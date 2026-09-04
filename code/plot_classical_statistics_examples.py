"""Plot observed price windows that illustrate the classical statistics in Section 2.1."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "price.csv"


def load_window(frame, series, start, end):
    window = frame.loc[
        frame["observation_date"].between(start, end),
        ["observation_date", series],
    ].dropna()
    window = window.rename(columns={series: "price"}).reset_index(drop=True)
    if len(window) != 80:
        raise ValueError(f"Expected 80 observations for {series}, found {len(window)}")
    return window


def maximum_drawup(values):
    best = (-np.inf, 0, 1)
    for start in range(len(values) - 1):
        end = start + 1 + np.argmax(values[start + 1 :] - values[start])
        candidate = values[end] - values[start]
        if candidate > best[0]:
            best = (candidate, start, end)
    return best


def maximum_drawdown(values):
    best = (-np.inf, 0, 1)
    for start in range(len(values) - 1):
        end = start + 1 + np.argmax(values[start] - values[start + 1 :])
        candidate = values[start] - values[end]
        if candidate > best[0]:
            best = (candidate, start, end)
    return best


def slope_change(values):
    time = np.arange(1, len(values) + 1, dtype=float)
    best = (-np.inf, None, None, None, None, None)
    for split in range(3, len(values) - 1):
        left_time, right_time = time[:split], time[split:]
        left_values, right_values = values[:split], values[split:]
        left_slope, left_intercept = np.polyfit(left_time, left_values, 1)
        right_slope, right_intercept = np.polyfit(right_time, right_values, 1)
        change = abs(left_slope - right_slope)
        if change > best[0]:
            best = (
                change,
                split,
                left_slope,
                left_intercept,
                right_slope,
                right_intercept,
            )
    return best


def event_observation_bounds(dates, start, end):
    """Return plot coordinates for an event interval on an observation axis."""
    date_index = pd.DatetimeIndex(dates)
    start_index = date_index.searchsorted(pd.Timestamp(start), side="left")
    end_index = date_index.searchsorted(pd.Timestamp(end), side="right") - 1
    start_index = int(np.clip(start_index, 0, len(date_index) - 1))
    end_index = int(np.clip(end_index, start_index, len(date_index) - 1))
    observation = np.arange(1, len(date_index) + 1, dtype=float)
    return (
        observation[start_index] - 0.5,
        observation[end_index] + 0.5,
    )


def style_axis(axis, dates):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="y", color="#d9d9d9", linewidth=0.5, alpha=0.75)
    axis.tick_params(axis="both", labelsize=7.5, length=3)
    dates = pd.DatetimeIndex(dates)
    month_starts = np.r_[True, dates.month[1:] != dates.month[:-1]]
    tick_positions = np.flatnonzero(month_starts) + 1
    axis.set_xticks(tick_positions)
    axis.set_xticklabels([dates[index - 1].strftime("%b\n%Y") for index in tick_positions])
    axis.set_xlabel("Date", fontsize=8)


def main():
    data = pd.read_csv(DATA_PATH, parse_dates=["observation_date"])
    spike = load_window(data, "DHHNGSP", "2020-12-01", "2021-03-26")
    explosive = load_window(data, "DCOILBRENTEU", "1990-05-01", "1990-08-21")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.unicode_minus": False,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.18), constrained_layout=True)

    # Panel (a) isolates the pathwise level and directional statistics.
    ax = axes[0]
    dates = spike["observation_date"].to_numpy()
    values = spike["price"].to_numpy(float)
    observation = np.arange(1, len(values) + 1, dtype=float)
    event_left, event_right = event_observation_bounds(
        dates, "2021-02-13", "2021-02-19"
    )
    ax.axvspan(
        event_left,
        event_right,
        color="#9aa7b5",
        alpha=0.16,
        linewidth=0,
        zorder=0,
    )
    ax.plot(observation, values, color="#222222", linewidth=1.35, zorder=3)

    range_value = values.max() - values.min()
    drawup, up_start, up_end = maximum_drawup(values)
    drawdown, down_start, down_end = maximum_drawdown(values)

    # Range is independent of temporal order.
    range_x = 85.0
    ax.annotate(
        "",
        xy=(range_x, values.max()),
        xytext=(range_x, values.min()),
        arrowprops=dict(arrowstyle="<->", color="#3d5a80", lw=1.0),
    )
    ax.text(
        range_x + 1.0,
        (values.max() + values.min()) / 2,
        rf"range $={range_value:.2f}$",
        fontsize=7.2,
        color="#3d5a80",
        rotation=90,
        va="center",
    )

    # Directional movements use their chronological start and end points.
    ax.annotate(
        "",
        xy=(observation[up_end], values[up_end]),
        xytext=(observation[up_start], values[up_start]),
        arrowprops=dict(
            arrowstyle="->",
            color="#2a7f62",
            lw=1.0,
            connectionstyle="arc3,rad=-0.08",
        ),
    )
    ax.text(11.5, 9.0, rf"drawup $D^+={drawup:.2f}$", fontsize=7.2, color="#2a7f62", rotation=25)
    ax.annotate(
        "",
        xy=(observation[down_end], values[down_end]),
        xytext=(observation[down_start], values[down_start]),
        arrowprops=dict(
            arrowstyle="->",
            color="#b24a3b",
            lw=1.0,
            linestyle="--",
            connectionstyle="arc3,rad=-0.08",
        ),
    )
    ax.text(
        59.0,
        21.5,
        rf"drawdown $D^-={drawdown:.2f}$",
        fontsize=7.2,
        color="#b24a3b",
        rotation=-64,
        rotation_mode="anchor",
        ha="left",
        va="top",
    )
    ax.annotate(
        "Winter Storm Uri\n13--19 Feb 2021",
        xy=((event_left + event_right) / 2, 24.0),
        xytext=(27.0, 24.8),
        fontsize=6.9,
        color="#59636f",
        ha="left",
        va="top",
        arrowprops=dict(arrowstyle="->", color="#6f7782", lw=0.8),
    )

    ax.set_title("(a) Range and directional movements\nHenry Hub natural gas, Dec 2020--Mar 2021", fontsize=8.5, pad=6)
    ax.set_ylabel("Dollars per million BTU", fontsize=8)
    ax.set_ylim(0.8, 26.0)
    ax.set_xlim(-1.5, 90.5)
    style_axis(ax, dates)

    # Panel (b) combines a volatility burst, an interior slope change, and a
    # positive no-intercept AR statistic in one observed window.
    ax = axes[1]
    dates = explosive["observation_date"].to_numpy()
    values = explosive["price"].to_numpy(float)
    observation = np.arange(1, len(values) + 1, dtype=float)
    increments = np.diff(values)
    realised_volatility = np.sum(increments**2)
    slope_value, split, left_slope, left_intercept, right_slope, right_intercept = slope_change(values)
    ar_score = np.sum(values[:-1] * increments)
    event_left, event_right = event_observation_bounds(
        dates, "1990-08-02", "1990-08-21"
    )
    ax.axvspan(
        event_left,
        event_right,
        color="#9aa7b5",
        alpha=0.16,
        linewidth=0,
        zorder=0,
    )
    ax.plot(observation, values, color="#222222", linewidth=1.35, zorder=3)

    # The event band begins when Iraq invaded Kuwait and continues to the end
    # of the plotted window.
    ax.text(
        event_left + 4.5,
        30.2,
        "Iraq invades Kuwait\n2 Aug 1990",
        fontsize=6.9,
        color="#59636f",
        ha="right",
        va="top",
    )

    # Plot the two fitted lines associated with the maximising split point.
    left_fit = left_slope * observation[:split] + left_intercept
    right_fit = right_slope * observation[split:] + right_intercept
    ax.plot(observation[:split], left_fit, color="#c07a00", linewidth=1.05, linestyle=(0, (4, 2)), zorder=2)
    ax.plot(observation[split:], right_fit, color="#c07a00", linewidth=1.05, linestyle=(0, (4, 2)), zorder=2)
    split_observation = observation[split]
    ax.axvline(split_observation, color="#c07a00", linewidth=0.8, linestyle=":", zorder=1)
    ax.text(13.5, 13.1, rf"fitted slopes give $S={slope_value:.3f}$", fontsize=7.2, color="#9a6200")

    ax.text(
        0.04,
        0.94,
        rf"realised volatility $V={realised_volatility:.2f}$"
        + "\n"
        + "autoregressive explosiveness"
        + "\n"
        + rf"$S_{{AR}}={ar_score:.2f}>0$",
        transform=ax.transAxes,
        fontsize=7.7,
        va="top",
        bbox=dict(facecolor="white", edgecolor="#bdbdbd", linewidth=0.5, alpha=0.9, pad=2.2),
    )
    ax.set_title("(b) Volatility, slope change and AR explosiveness\nBrent crude oil, May--Aug 1990", fontsize=8.5, pad=6)
    ax.set_ylabel("Dollars per barrel", fontsize=8)
    ax.set_ylim(11.8, 31.2)
    ax.set_xlim(-1.5, 81.5)
    style_axis(ax, dates)

    plt.show()


if __name__ == "__main__":
    main()
