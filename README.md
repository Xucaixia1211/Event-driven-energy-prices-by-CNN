# A convolutional framework for detecting event-driven dynamics in energy price series

This repository contains the code, frozen input data and reported numerical outputs for:

> Caixia Xu and Piotr Fryzlewicz, *A convolutional framework for detecting event-driven dynamics in energy price series*.

The current preprint is available on [arXiv:2609.00402](https://arxiv.org/abs/2609.00402). The repository is organised as a compact reproducibility release rather than as a record of exploratory development.

## Repository contents

- `code/` contains the empirical and simulation notebooks and the plotting code for the figures reported in the paper.
- `data/` contains the frozen FRED export, its metadata and the curated event table used by the empirical notebook.
- `figures/` contains the empirical and simulation figures reported in the paper.
- `results/` contains the reported numerical tables and the raw Monte Carlo output used to construct them.
- `DATA_DICTIONARY.md` defines the fields in the frozen data and principal result files.

## Shared model framework

`code/models/general_cnn.py` implements the common architecture: one or more feature branches are evaluated in parallel, their pooled outputs are concatenated, and a common head produces the final score or class logits. The two analyses select different instances of this framework through parameters rather than redefining its structure.

- `code/models/hierarchical_classifiers.py` builds the trainable three-block CNN used at each stage of the hierarchical application.
- `code/models/classical_statistic_classifiers.py` builds the fixed CNN classifiers based on range, drawup, drawdown and slope and the realised-volatility and AR approximations used in the simulations.
- `code/models/benchmark_classifiers.py` contains the ResNet-type CNN, MLP, LSTM and FT-Transformer architectures used only in the empirical comparison.

Loss functions, sample weights, data splitting and optimisation remain in the analysis notebooks because they describe estimation rather than the CNN architecture.

## Environment

The reported results were produced with Python 3.12.7 on arm64 macOS. Install the recorded direct dependencies in a fresh environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch may require a platform-specific installation command on systems with CUDA. See the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) if the command above is not suitable for the target machine.

## Reproduction

`run_all.sh` is an optional command-line wrapper that executes the notebooks from a clean kernel without writing their displayed outputs to disk. The same analyses can instead be reproduced by opening the notebooks and running all cells in order.

Run the empirical analysis with:

```bash
./run_all.sh empirical
```

The simulation notebook defaults to five Monte Carlo repetitions for a quick interactive run. The full paper reproduction uses 500 repetitions:

```bash
./run_all.sh simulation
```

A short structural test of the simulation uses five repetitions:

```bash
./run_all.sh simulation-smoke
```

The notebooks display their tables and figures in their cells but do not create directories or save generated artifacts. The reported outputs are already frozen under `results/application/`, `results/simulation/`, `figures/application/` and `figures/simulation/`. Full execution can take substantial time on a CPU-only machine, particularly for the empirical benchmark comparison and the 500-repetition simulation.

## Reported outputs

The following table maps the principal paper results to their generating code and frozen outputs.

| Paper result | Generating source | Frozen output |
|---|---|---|
| Classical-statistic examples | `code/plot_classical_statistics_examples.py` | `figures/classical_statistics_examples.pdf` |
| CNN architecture diagram | Manually prepared diagram | `figures/cnn1d_classifier_flowchart.pdf` |
| Exact CNN representation checks | `code/simulation/theory_validation_simulation.ipynb` | `results/simulation/representation/exact_representation_results.csv` |
| Realised-volatility and AR approximation checks | `code/simulation/theory_validation_simulation.ipynb` | `results/simulation/representation/approximation_results.csv` |
| Approximation-error figure | `code/simulation/theory_validation_simulation.ipynb` | `figures/simulation/approximation_error_by_depth.png` |
| Oracle-comparison simulation | `code/simulation/theory_validation_simulation.ipynb` | `results/simulation/theorem9_oracle/T40_R500_D9/` and `figures/simulation/theorem9_oracle/T40_R500_D9/` |
| Stage 1 binary results | `code/application/empirical_analysis.ipynb` | `results/application/stage1_binary_metrics.csv` |
| Stage 1 benchmark comparison | `code/application/empirical_analysis.ipynb` | `results/application/paper_baseline_comparison_*.csv` |
| Hierarchical component and joint results | `code/application/empirical_analysis.ipynb` | `results/application/hierarchical_metrics.csv` and `results/application/multiclass_confusion_percent.csv` |
| Empirical examples and independent case study | `code/application/empirical_analysis.ipynb` | `figures/application/` |

The long-form benchmark file `paper_baseline_comparison_components_6commodities.csv` contains the commodity-specific values used to calculate the reported unweighted means and standard deviations. The oracle-comparison directory contains the full 500-repetition output and its summary.

## Reproducibility details

- The empirical window length is fixed at 80, matching the reported analysis.
- The empirical and simulation notebooks use global random seed 40. Stage 1 downsampling uses seed 42.
- The empirical notebook trains the models from scratch; no unpublished checkpoint is required.
- The stored result files are the frozen outputs used for the manuscript. Small floating-point differences can arise across PyTorch versions, hardware backends and operating systems.
- The notebooks were cleared of execution output before release to remove machine-specific paths. The reported numerical results remain available under the `results/` directories.

## Data

The price series are public daily observations downloaded from [Federal Reserve Economic Data (FRED)](https://fred.stlouisfed.org/). The included `data/price.csv` contains only the six series used in the paper so that later source revisions do not alter the reproduction. Details of the series and data terms are provided in `data/README.md` and `data/FRED_README.txt`.

The event table was curated for this study. It records the event intervals, key dates, categories and source information used by the labelling procedure. Different commodities associated with the same historical episode are represented by separate records, in accordance with the paper.

## Licence

The original code and documentation in this repository are released under the MIT License. The FRED data remain subject to the terms of their original providers, and this repository does not relicense third-party data or linked source material. See `data/README.md`.
