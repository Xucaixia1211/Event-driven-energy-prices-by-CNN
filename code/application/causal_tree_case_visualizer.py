"""Plot representative correct and confused cases from the causal CNN tree."""

import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_CASES = {
    "correct_weather": {
        "true_label": 1,
        "pred_label": 1,
        "title": "Correctly classified Weather event",
    },
    "correct_geopolitical": {
        "true_label": 2,
        "pred_label": 2,
        "title": "Correctly classified Geopolitical event",
    },
    "correct_supply_macro": {
        "true_label": 3,
        "pred_label": 3,
        "title": "Correctly classified Supply-financial event",
    },
    "supply_macro_as_geopolitical": {
        "true_label": 3,
        "pred_label": 2,
        "title": "Supply-financial event classified as Geopolitical",
    },
    "geopolitical_as_supply_macro": {
        "true_label": 2,
        "pred_label": 3,
        "title": "Geopolitical event classified as Supply-financial",
    },
}


DEFAULT_LABEL_NAMES = {
    0: "No event",
    1: "Weather / natural hazard",
    2: "Geopolitical / security",
    3: "Supply-financial",
}


PRICE_UNITS = {
    "DCOILBRENTEU": "Dollars per Barrel",
    "DHHNGSP": "Dollars per Million BTU",
    "DDFUELUSGULF": "Dollars per Gallon",
    "DGASNYH": "Dollars per Gallon",
    "DJFUELUSGULF": "Dollars per Gallon",
    "DRGASLA": "Dollars per Gallon",
}


def _window_candidates(prediction_df, true_label, pred_label):
    candidates = prediction_df.loc[
        (prediction_df["true_label"].astype(int) == int(true_label))
        & (prediction_df["pred_label"].astype(int) == int(pred_label))
    ].copy()
    if candidates.empty:
        return candidates

    candidates["_event_anchor"] = candidates.get(
        "event_date_in_window", pd.Series(False, index=candidates.index)
    ).fillna(False).astype(int)
    candidates["_overlap"] = pd.to_numeric(
        candidates.get("event_overlap_ratio", 0.0), errors="coerce"
    ).fillna(0.0)
    candidates["_event_key"] = candidates.get(
        "event_id", pd.Series(index=candidates.index, dtype=object)
    ).astype(object)
    missing_event = candidates["_event_key"].isna()
    candidates.loc[missing_event, "_event_key"] = (
        "sample_" + candidates.loc[missing_event, "sample_index"].astype(str)
    )

    candidates = candidates.sort_values(
        ["commodity", "_event_key", "_event_anchor", "_overlap", "sample_index"],
        ascending=[True, True, False, False, True],
    )
    candidates = candidates.drop_duplicates(["commodity", "_event_key"], keep="first")
    return candidates.sort_values(
        ["commodity", "window_start", "sample_index"]
    ).reset_index(drop=True)


def _events_for_row(row, events_by_commodity):
    commodity = row["commodity"]
    window_start = pd.Timestamp(row["window_start"])
    window_end = pd.Timestamp(row["window_end"])
    event_id = row.get("event_id")
    exact_match = pd.notna(event_id)

    selected_events = []
    for event in events_by_commodity.get(commodity, []):
        if exact_match and str(event.get("event_id")) != str(event_id):
            continue
        event_start = pd.Timestamp(event["start_date"])
        event_end = pd.Timestamp(event["end_date"])
        if window_start <= event_end and window_end >= event_start:
            selected_events.append(event)

    if selected_events or not exact_match:
        return selected_events

    # Metadata and event tables can occasionally use different event-id dtypes.
    for event in events_by_commodity.get(commodity, []):
        event_start = pd.Timestamp(event["start_date"])
        event_end = pd.Timestamp(event["end_date"])
        if window_start <= event_end and window_end >= event_start:
            selected_events.append(event)
    return selected_events


def _format_probability_path(row):
    probability_columns = [
        ("stage1_event_probability", "event"),
        ("node1_weather_probability", "weather"),
        ("node2_geopolitical_probability", "geopolitical"),
    ]
    path_parts = []
    for column, label in probability_columns:
        value = row.get(column, np.nan)
        if pd.notna(value):
            path_parts.append(f"p({label})={float(value):.3f}")
    return " | ".join(path_parts)


def render_causal_tree_examples(
    prediction_df,
    price_df,
    events_by_commodity,
    cases=None,
    sample_numbers=None,
    annotation_options=None,
    label_names=None,
    plot_length_points=120,
    figsize=(8.8, 4.9),
    dpi=220,
    show=True,
):
    """Render one distinct-event example per requested causal-tree case."""
    required_columns = {
        "sample_index",
        "commodity",
        "window_start",
        "window_end",
        "true_label",
        "pred_label",
    }
    missing_columns = sorted(required_columns.difference(prediction_df.columns))
    if missing_columns:
        raise ValueError(f"Prediction table is missing columns: {missing_columns}")

    cases = DEFAULT_CASES if cases is None else cases
    sample_numbers = {} if sample_numbers is None else sample_numbers
    annotation_options = {} if annotation_options is None else annotation_options
    label_names = DEFAULT_LABEL_NAMES if label_names is None else label_names
    summary_rows = []
    for case_name, case in cases.items():
        candidates = _window_candidates(
            prediction_df, case["true_label"], case["pred_label"]
        )
        sample_number = int(sample_numbers.get(case_name, 1))
        if candidates.empty:
            raise ValueError(f"No prediction windows are available for {case_name!r}.")
        if not 1 <= sample_number <= len(candidates):
            raise IndexError(
                f"{case_name} sample number must be between 1 and {len(candidates)} "
                "distinct commodity-event candidates."
            )

        selected = candidates.iloc[sample_number - 1]
        commodity = str(selected["commodity"])
        window_start = pd.Timestamp(selected["window_start"])
        window_end = pd.Timestamp(selected["window_end"])

        raw_series = price_df[commodity].dropna().copy().sort_index()
        raw_series.index = pd.to_datetime(raw_series.index)
        selected_window = raw_series.loc[window_start:window_end]
        if selected_window.empty:
            raise ValueError(
                f"No raw prices found for {commodity} from {window_start} to {window_end}."
            )

        selected_start_pos = int(raw_series.index.searchsorted(window_start, side="left"))
        selected_end_pos = int(raw_series.index.searchsorted(window_end, side="right"))
        selected_length = selected_end_pos - selected_start_pos
        context_length = max(int(plot_length_points), selected_length)
        extra_points = context_length - selected_length
        plot_start_pos = max(0, selected_start_pos - extra_points // 2)
        plot_end_pos = min(
            len(raw_series), selected_end_pos + extra_points - extra_points // 2
        )
        visible_prices = raw_series.iloc[plot_start_pos:plot_end_pos]
        plot_start = visible_prices.index[0]
        plot_end = visible_prices.index[-1]

        options = annotation_options.get(case_name, {})
        case_figsize = tuple(options.get("figsize", figsize))
        case_dpi = int(options.get("dpi", dpi))
        fig, ax = plt.subplots(
            figsize=case_figsize,
            dpi=case_dpi,
            constrained_layout=True,
        )
        ax.plot(
            visible_prices.index,
            visible_prices.values,
            color="#263746",
            linewidth=1.4,
            label="Price",
        )
        ax.plot(
            selected_window.index,
            selected_window.values,
            color="#dc3545",
            linewidth=3.8,
            alpha=0.88,
            solid_capstyle="round",
            label="Selected window",
        )

        events = _events_for_row(selected, events_by_commodity)
        event_names = []
        wrap_width = int(options.get("wrap_width", 34))

        for event in events:
            event_start_raw = pd.Timestamp(event["start_date"])
            event_end_raw = pd.Timestamp(event["end_date"])
            event_start = max(plot_start, event_start_raw)
            event_end = min(plot_end, event_end_raw)
            if event_start > event_end:
                continue

            event_name = str(event.get("subcategory") or event.get("event_id") or "Event")
            event_names.append(event_name)
            event_legend_label = textwrap.fill(event_name, width=wrap_width)
            event_legend_label += (
                f"\n{event_start_raw:%Y-%m-%d} to {event_end_raw:%Y-%m-%d}"
            )
            ax.axvspan(
                event_start,
                event_end,
                color="#16a34a",
                alpha=0.13,
                label=event_legend_label,
            )
            ax.axvline(
                event_start,
                color="#16a34a",
                linestyle="--",
                linewidth=0.85,
                alpha=0.65,
            )
            ax.axvline(
                event_end,
                color="#16a34a",
                linestyle="--",
                linewidth=0.85,
                alpha=0.65,
            )

        true_name = label_names.get(int(selected["true_label"]), str(selected["true_label"]))
        pred_name = label_names.get(int(selected["pred_label"]), str(selected["pred_label"]))
        probability_path = _format_probability_path(selected)
        ax.set_title(
            f"{case['title']}\n{probability_path}",
            fontsize=11,
        )
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel(PRICE_UNITS.get(commodity, "Price"), fontsize=11)
        ax.tick_params(axis="both", labelsize=10)
        ax.grid(True, color="#e5e7eb", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc=options.get("legend_loc", "best"), fontsize=10, frameon=True)

        if show:
            plt.show()
        else:
            plt.close(fig)

        summary_rows.append(
            {
                "case": case_name,
                "distinct_event_candidates": len(candidates),
                "sample_number": sample_number,
                "sample_index": int(selected["sample_index"]),
                "commodity": commodity,
                "event_id": selected.get("event_id"),
                "event": "; ".join(dict.fromkeys(event_names)),
                "window_start": window_start,
                "window_end": window_end,
                "true_class": true_name,
                "predicted_class": pred_name,
                "stage1_event_probability": selected.get("stage1_event_probability"),
                "node1_weather_probability": selected.get("node1_weather_probability"),
                "node2_geopolitical_probability": selected.get(
                    "node2_geopolitical_probability"
                ),
            }
        )

    return pd.DataFrame(summary_rows)
