# Data dictionary

## Energy-price data

`data/price.csv` contains one observation date and the six daily
energy-price series used in the paper. The series were downloaded from FRED,
and missing values retain the source encoding.

| Field | Meaning |
|---|---|
| `observation_date` | Calendar date of the observation. |
| Remaining fields | FRED series identifiers; each value is the reported price for that date and series. |

## Curated event data

`data/events.xlsx` has one row for each
commodity and historical event pairing. The same historical episode associated
with different commodities is therefore represented by separate rows.

| Field | Meaning |
|---|---|
| `event_id` | Event identifier used during curation. |
| `commodity` | FRED series associated with the event. |
| `category` | Event family used by the hierarchical classifier: weather and natural hazard, geopolitical and security, or the combined supply, policy, macroeconomic and financial family. |
| `causal_category` | Causal grouping recorded during event-table preparation. |
| `subcategory` | More detailed event description. |
| `start_date`, `end_date` | Inclusive event interval used for window labelling. |
| `event_date` | Principal date associated with the event. |
| `region`, `intensity`, `description` | Curated event characteristics. |
| `source`, `source_url` | Source name and available public URL. |
| `confidence` | Curated confidence assessment. |
| `event_structure_*` | Alternative event-structure labels and their rule-based metadata retained from data preparation. |

## Application results

The CSV files in `results/application/` contain the values reported in the
empirical tables. Standard classification fields have their usual meanings.
`eval_n` is the denominator used for a component metric, while `routed_n` is
the number of observations routed to that component. `purity` is the share of
routed observations that belong within the component's intended scope, and
`effective_accuracy` is the number of correct component decisions divided by
all routed observations.

The long-form benchmark file records one row per commodity and method. The
average and standard-deviation files aggregate those six commodity-specific
rows without weighting them by the number of windows.

## Simulation results

`results/simulation/representation/` stores the sampled exact-representation
and approximation checks. For the approximation table, `max_abs_error` and
`mean_abs_error` summarize errors on the sampled compact domain, and
`classification_agreement` records agreement outside the specified threshold
margin.

`results/simulation/theorem9_oracle/T40_R500_D9/theorem9_results_raw.csv`
contains one row per scenario, replication and training-sample size. Comparator
columns report the fixed slope, realised-volatility and autoregressive rules.
The oracle column gives the lowest test error among those three rules, the
selector columns describe the rule chosen on the training sample, and the
learned-model columns report the fitted joint classifier. The corresponding
summary file records the means and standard deviations used in the paper.
