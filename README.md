# CSE5104 Project 1: Concrete Compressive Strength

Linear-regression analysis of the UCI Concrete Compressive Strength data set
(1,030 mixes, 8 ingredients/age → strength in MPa): a from-scratch gradient
descent optimiser, feature scaling and log transforms, univariate vs.
multivariate coefficients, and OLS significance tests.

## Team and division of work

| Member | Role | Code | Report sections |
|---|---|---|---|
| Haoyu Zhu (A) | Optimizer and feature scaling | `gd.py`, `experiments/a_scaling.py` | Part 2: Problem Context and Data Overview. Part 3: Algorithmic Design and Implementation Details; Feature Scaling and Gradient Descent Dynamics |
| James Chen (B) | Data layer and feature transformations | `data.py`, `experiments/b_log_transform.py`, `experiments/b_univariate.py`, `tests/test_data.py` | Part 2: Model Reliability and Analytical Findings. Part 3: Linear vs. Non-Linear Feature Transformations; Univariate vs. Multivariate Model Dynamics |
| Jian Chen (C) | Statistical inference and integration | `metrics.py`, `inference.py`, `result_schema.py`, `aggregate.py`, `run_all.py`, `config.py`, `experiments/c_inference.py`, the other unit tests, this README | Part 2: Executive Summary; Strategic Recommendations and Implementation Plan. Part 3: Statistical Inference and Regression Analysis; Code and Reproducibility Access; final formatting of the report |

The letters A, B and C are used as owner tags throughout the code (for example in
`run_all.py` and in result-file names) and in the tables below.

## Repository layout

| Path | Owner | Purpose |
|---|---|---|
| `config.py` | C | Locked parameters and paths (data file, test rows, learning-rate grid, …) |
| `data/` | B | The data set. Experiments read `Concrete_Data.csv`, an exact export of the original UCI file `Concrete_Data.xls` |
| `data.py` | B | Load, split, transform, recover coefficients |
| `gd.py` | A | Gradient descent on the MSE objective |
| `metrics.py` | C | MSE and R² |
| `inference.py` | C | Standard errors, t-statistics, p-values (statsmodels + hand-written check) |
| `result_schema.py` | C | Format of `results/<exp_id>.json`; `make_result` / `write_result` |
| `aggregate.py` | C | Builds the report tables and figures from `results/` |
| `run_all.py` | C | Runs every registered experiment, then aggregates |
| `experiments/` | A, B, C | One script per analysis: `a_scaling.py` (A), `b_log_transform.py` and `b_univariate.py` (B), `c_inference.py` (C) |
| `results/` | generated | `results/<exp_id>.json` from B's and C's experiments; CSV tables and figures from A's experiment in `results/a/` |
| `figures/`, `report/tables/` | generated | Figures from B's experiment and `aggregate.py`; report tables built by `aggregate.py` |
| `tests/` | C, B | Unit tests (standard-library `unittest`); `test_data.py` by B |

## Setup

Requires Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` on macOS/Linux
pip install -r requirements.txt
```

## Reproducing the results

Run everything (experiments in a fresh interpreter each, then aggregation):

```bash
python run_all.py
```

Useful options: `--list` (show registered experiments), `--only A B C`,
`--aggregate-only`, `--strict` (fail if a registered script is missing).

Every experiment reads the data file set in `config.DATA_PATH`. If that file is
missing, `run_all.py` stops before running anything (exit status 2) and names the
data files it did find; individual experiments accept `--data-path FILE` instead.

Individual steps, all from the repository root:

```bash
python -m experiments.c_inference     # one experiment
python aggregate.py                   # rebuild tables/figures from results/
python -m unittest discover -s tests -t .
```

## Report tables and figures

Every table and figure in the report comes from a file produced by the code:

| Report item | Produced by | File |
|---|---|---|
| Table 1: descriptive statistics (training set) | `experiments/a_scaling.py` | `results/a/descriptive_stats_train.csv` |
| Table 2: raw vs. standardized coefficients | `experiments/a_scaling.py` | `results/a/table2_ols_raw_vs_standardized.csv` |
| Table 3: the three optimal models | `experiments/c_inference.py`, then `aggregate.py` | `report/tables/optimal_models_summary` |
| Table 4: univariate vs. multivariate coefficients | `experiments/b_univariate.py`, then `aggregate.py` | `report/tables/univariate_vs_multivariate_coefficients` |
| Table 5: p-values by configuration | `experiments/c_inference.py`, then `aggregate.py` | `report/tables/pvalues_by_configuration` |
| Figures 1–2: cement and age vs. strength | `experiments/a_scaling.py` | `results/a/cement_vs_strength.png`, `results/a/age_vs_strength.png` |
| Figure 3: loss with raw vs. standardized features | `experiments/a_scaling.py` | `results/a/raw_vs_standardized_loss.png` |
| Figure 4: loss with log-transformed features | `experiments/b_log_transform.py` | `figures/b_log1p_loss.png` |
| Figure 5: p-values by configuration | `aggregate.py` | `figures/pvalues_by_configuration.png` |

`aggregate.py` also writes `report/tables/univariate_gd_training`,
`report/tables/significance_vs_univariate_training` and
`report/tables/all_experiments` (every run's settings and metrics, for
cross-checking). Each table is written twice: a CSV with full precision and a
Markdown copy.

## Data partition

Rows 501–630 of the data file (1-indexed) are the 130-row **test set**, and the
other 900 rows the **training set** (`config.TEST_ROWS`). Split first, then
transform: scaling and log parameters are fitted on the training rows only and
reused for the test rows. No model parameter is ever fitted on test data.

## Interfaces

Shared conventions: `X` is the feature matrix **without** an intercept column
(a 1-D array means one feature), and `theta` has `n_features + 1` entries with
the **intercept first**.

| Owner | Function | Returns |
|---|---|---|
| A | `gradient_descent(X, y, alpha, n_iters, tol=1e-8)` | `{"theta", "loss_history", "converged_iter"}` |
| B | `load_data(path)` | `(X, y, feature_names)`, all 1,030 rows in file order |
| B | `split(X, y)` | `(X_train, X_test, y_train, y_test)` using the partition above |
| B | `transform(X, mode, params=None)` | `(X_out, params)`; `mode` ∈ `raw`, `standardize`, `log1p`; pass the training `params` when transforming the test set |
| B | `recover_coef(theta, params)` | `theta` on the original feature scale |
| C | `evaluate(X, y, theta)` | `{"mse", "r2"}` with R² = 1 − MSE / Var(y) |
| C | `evaluate_split(X_train, y_train, X_test, y_test, theta)` | `{"train_mse", "test_mse", "train_r2", "test_r2"}` |
| C | `ols_inference(X, y, theta)` | `{"se", "t_stat", "p_value"}`, aligned with `theta` |
| C | `fit_ols_statsmodels(X, y)` | `{"theta", "se", "t_stat", "p_value"}` |
| C | `collect(results_dir)` | `DataFrame` with one row per result file |

## Writing a result file

B's and C's experiments write one `results/<exp_id>.json` per run with
`result_schema.write_result`; A's experiment writes CSV tables and figures to
`results/a/`. The report quotes numbers only from these files and from the
tables `aggregate.py` builds from them. The field reference is in
`result_schema.py`. For example, from `experiments/b_log_transform.py`:

```python
from metrics import evaluate_split
from result_schema import fields_for_mode, make_result, write_result

result = gradient_descent(X_train, y_train, alpha=0.003, n_iters=5000)
record = make_result(
    exp_id="b_gd_log1p_a0p003",             # must start with the owner's letter
    owner="B",
    solver="gd",
    **fields_for_mode("log1p"),              # fills transform / scaling
    alpha=0.003, n_iters=5000, converged_iter=result["converged_iter"],
    theta=result["theta"],
    features=feature_names,                  # names for theta[1:]
    **evaluate_split(X_train, y_train, X_test, y_test, result["theta"]),
)
write_result(record)                         # -> results/b_gd_log1p_a0p003.json
```

For univariate models, set `features` to the single feature's name. That is how
`aggregate.py` finds them for the univariate and trend tables.

## Adding an experiment

1. Create `experiments/<owner>_<topic>.py`. It must run with
   `python -m experiments.<owner>_<topic>` from the repository root, write
   results with `write_result`, and save figures under `config.FIGURES_DIR`.
2. Add one line to `EXPERIMENTS` in `run_all.py`.

## Data source

I-Cheng Yeh, "Modeling of strength of high performance concrete using artificial
neural networks," *Cement and Concrete Research*, 28(12), 1797–1808 (1998).
UCI Machine Learning Repository:
<https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength>.
