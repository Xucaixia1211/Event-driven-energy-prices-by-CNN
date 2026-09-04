# Data provenance

## Energy prices

`price.csv` contains the six daily FRED energy-price series used in the paper. It was extracted from the frozen export created on 25 July 2026 from data updated through 22 July 2026. Its columns appear in the same order used by the reported empirical analysis:

| FRED series | Description |
|---|---|
| `DCOILBRENTEU` | Brent crude oil price, Europe |
| `DGASNYH` | Conventional gasoline price, New York Harbor |
| `DHHNGSP` | Henry Hub natural gas spot price |
| `DDFUELUSGULF` | Ultra-low-sulfur No. 2 diesel fuel price, U.S. Gulf Coast |
| `DRGASLA` | Regular gasoline price, Los Angeles |
| `DJFUELUSGULF` | Kerosene-type jet fuel price, U.S. Gulf Coast |

The source is [Federal Reserve Economic Data](https://fred.stlouisfed.org/), with the underlying series supplied by the U.S. Energy Information Administration. `FRED_README.txt` contains the corresponding metadata extracted from the original export. Users remain responsible for compliance with the [FRED terms of use](https://fred.stlouisfed.org/legal/).

SHA-256 for the frozen file:

```text
22c0f479491a2de756586d053701fa2224e9a3d0f68692e7629efd13b5fc7691  price.csv
```

## Curated events

`events.xlsx` is the event table used directly by both stages of the empirical analysis. It contains 86 commodity-event records and retains the source names and available source URLs used during curation. Its `category` column records the three event families used in the paper: `weather_natural_hazard`, `geopolitical_security` and `supply_policy_macro_financial`. The acute/extended event-structure label is recorded separately in `event_structure_category`.

SHA-256 for the frozen event table:

```text
ee44cedf7b39b6ce5291fc9f30debb2a88bd2d691192adf1d62556f81d4fc9f0  events.xlsx
```

The event table is supplied to reproduce this study. Copyright in linked news, institutional descriptions and third-party materials remains with the original providers.
