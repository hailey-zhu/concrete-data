from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

import config
from result_schema import ALL_FIELDS, METRIC_FIELDS, ResultSchemaError, config_label, read_result

__all__ = [
    "MissingResultError", "build_report", "coefficient_comparison_table", "collect",
    "format_pvalue", "overview_table", "plot_pvalues", "pvalue_table", "save_table",
    "summary_table", "to_markdown", "trend_table", "univariate_table",
]

logger = logging.getLogger(__name__)

PathLike = str | os.PathLike[str]
#: A float format: a format spec such as ``".4f"`` or a function returning the cell text.
FloatFormat = str | Callable[[float], str]

#: exp_ids of the three optimal (OLS) models, in report order.
DEFAULT_OLS_IDS: tuple[str, ...] = tuple(config.OLS_EXP_IDS.values())

#: Optimal multivariate model used for the coefficient comparison, as written
#: by experiments/b_univariate.py.
DEFAULT_MULTIVARIATE_EXP_ID: str = "b_sklearn_multivariate_raw"

METRIC_LABELS: dict[str, str] = {
    "train_mse": "Training MSE",
    "test_mse": "Testing MSE",
    "train_r2": "Training R²",
    "test_r2": "Testing R²",
}
_OVERVIEW_LABELS: dict[str, str] = {
    "owner": "Owner",
    "config": "Configuration",
    "solver": "Solver",
    "alpha": "Learning rate",
    "n_iters": "Max iterations",
    "converged_iter": "Converged at",
    **METRIC_LABELS,
}
_NUMBER_FORMATS: dict[str, str] = {
    "Learning rate": ".3g",
    "Training MSE": ".2f",
    "Testing MSE": ".2f",
    "Training R²": ".4f",
    "Testing R²": ".4f",
}

# Chart styling. A configuration keeps the same colour in every figure.
CONFIG_COLORS: dict[str, str] = {
    "Untransformed": "#2a78d6",
    "Standardized": "#eb6834",
    "Normalized": "#eb6834",
    "Log-transformed": "#1baf7a",
}
_EXTRA_COLORS: tuple[str, ...] = ("#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
_SURFACE, _INK, _INK_SECONDARY = "#fcfcfb", "#0b0b0b", "#52514e"
_GRIDLINE, _BASELINE = "#e1e0d9", "#c3c2b7"
_MIN_PVALUE = 1e-300  # p-values below this have underflowed; shown as "<1e-300"


class MissingResultError(LookupError):
    """A table needs an experiment whose result file does not exist yet."""


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def collect(results_dir: PathLike = config.RESULTS_DIR, *, strict: bool = True) -> pd.DataFrame:
    """Read every ``<results_dir>/*.json`` result file into one table.

    Parameters
    ----------
    results_dir : path
        Directory holding the result files (sub-directories are ignored).
    strict : bool, default True
        If True, an invalid file raises :class:`ResultSchemaError`. If False,
        it is skipped with a warning.

    Returns
    -------
    DataFrame
        Indexed by ``exp_id`` (sorted), one column per schema field plus
        ``config`` (e.g. ``"Standardized"``). ``theta``, ``p_value`` and
        ``features`` hold Python lists. Empty if the directory has no files.

    Raises
    ------
    FileNotFoundError
        If ``results_dir`` does not exist.
    """
    directory = Path(results_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"results directory not found: {directory}")

    records = []
    for path in sorted(directory.glob("*.json")):
        try:
            records.append(read_result(path))
        except ResultSchemaError as exc:
            if strict:
                raise
            warnings.warn(f"skipping invalid result file: {exc}", stacklevel=2)

    frame = pd.DataFrame.from_records(records, columns=list(ALL_FIELDS))
    frame["config"] = [
        config_label(transform, scaling)
        for transform, scaling in zip(frame["transform"], frame["scaling"])
    ]
    for column in ("n_iters", "converged_iter"):
        frame[column] = frame[column].astype("Int64")
    return frame.set_index("exp_id").sort_index()


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def overview_table(results: pd.DataFrame) -> pd.DataFrame:
    """Settings and metrics of every experiment, so each member can check their numbers."""
    table = results[list(_OVERVIEW_LABELS)].rename(columns=_OVERVIEW_LABELS)
    table.index.name = "Experiment"
    return table


def summary_table(results: pd.DataFrame,
                  exp_ids: Sequence[str] = DEFAULT_OLS_IDS) -> pd.DataFrame:
    """Training/Testing MSE and R² of the selected models, one row per configuration.

    By default the rows are the three optimal OLS models (untransformed,
    standardized, log-transformed) written by ``experiments/c_inference.py``.

    Raises
    ------
    MissingResultError
        If an ``exp_id`` has no result file.
    ValueError
        If two selected experiments share a configuration.
    """
    rows = _select(results, exp_ids)
    table = rows[list(METRIC_FIELDS)].rename(columns=METRIC_LABELS).astype(float)
    table.index = pd.Index(_unique_labels(rows), name="Model")
    return table


def pvalue_table(results: pd.DataFrame,
                 exp_ids: Sequence[str] = DEFAULT_OLS_IDS) -> pd.DataFrame:
    """p-value of every term (rows: ``Intercept`` then features) for each configuration (columns).

    Raises
    ------
    MissingResultError
        If an ``exp_id`` has no result file.
    ValueError
        If a selected experiment has no p-values or two share a configuration.
    """
    rows = _select(results, exp_ids)
    columns: dict[str, pd.Series] = {}
    term_order: list[str] = []
    for (exp_id, row), label in zip(rows.iterrows(), _unique_labels(rows)):
        if _is_missing(row["p_value"]):
            raise ValueError(f"experiment {exp_id!r} has no p-values")
        features = row["features"]
        if _is_missing(features):
            features = [f"x{i}" for i in range(1, len(row["theta"]))]
        terms = ["Intercept", *features]
        columns[label] = pd.Series(row["p_value"], index=terms, dtype=float)
        term_order.extend(term for term in terms if term not in term_order)

    table = pd.DataFrame(columns).reindex(term_order)
    table.index.name = "Term"
    return table


def univariate_table(results: pd.DataFrame) -> pd.DataFrame:
    """Best gradient-descent run (highest training R²) for each single-feature model.

    A run counts as univariate gradient descent when its ``features`` list has
    exactly one name and it used gradient descent (``solver == "gd"``, or no
    solver recorded but a learning rate present). Rows are sorted from easiest
    to hardest to train (highest to lowest training R²).
    """
    columns = ["Experiment", "Learning rate", "Converged at", "Training R²", "Testing R²"]
    is_univariate_gd = np.array([_is_univariate_gd(row) for _, row in results.iterrows()],
                                dtype=bool)
    runs = results[is_univariate_gd]
    if runs.empty:
        return pd.DataFrame(columns=columns, index=pd.Index([], name="Feature"))

    runs = runs.assign(Feature=[features[0] for features in runs["features"]])
    best = (runs.reset_index()
            .sort_values("train_r2", ascending=False, na_position="last")
            .drop_duplicates("Feature"))
    table = best.set_index("Feature")[["exp_id", "alpha", "converged_iter", "train_r2", "test_r2"]]
    table.columns = columns
    return table


def coefficient_comparison_table(
        results: pd.DataFrame,
        multivariate_exp_id: str = DEFAULT_MULTIVARIATE_EXP_ID,
        univariate_exp_ids: Sequence[str] | None = None) -> pd.DataFrame:
    """Each feature's univariate slope beside its weight in the multivariate model.

    Answers "are the univariate and multivariate coefficients identical?".
    By default the univariate models are every single-feature result fitted the
    same way as the multivariate one (same configuration and solver), i.e. the
    optimal fits rather than the gradient-descent runs. A feature with no
    univariate model keeps a missing value and is named in a warning.

    Rows follow the multivariate model's feature order.

    Raises
    ------
    MissingResultError
        If the multivariate result, or every univariate result, is missing.
    ValueError
        If the multivariate result has no feature names to match on.
    """
    multivariate = _select(results, [multivariate_exp_id]).iloc[0]
    features = multivariate["features"]
    if _is_missing(features):
        raise ValueError(
            f"{multivariate_exp_id!r} has no feature names, so its weights cannot be "
            "matched to the univariate models"
        )

    if univariate_exp_ids is None:
        matches = np.array([_matches_multivariate(row, multivariate)
                            for _, row in results.iterrows()], dtype=bool)
        rows = results[matches]
    else:
        rows = _select(results, univariate_exp_ids)
    if rows.empty:
        raise MissingResultError(
            f"no univariate results to compare with {multivariate_exp_id!r} (solver "
            f"{multivariate['solver']!r}, configuration {multivariate['config']!r}); "
            "run the experiment that produces them first"
        )

    slopes = {row["features"][0]: row["theta"][1] for _, row in rows.iterrows()}
    unmatched = [name for name in features if name not in slopes]
    if unmatched:
        logger.warning("No univariate model for: %s", ", ".join(unmatched))

    table = pd.DataFrame(
        {
            "Univariate coefficient": [slopes.get(name, float("nan")) for name in features],
            "Multivariate coefficient": list(multivariate["theta"][1:]),
        },
        index=pd.Index(features, name="Feature"),
    )
    table["Difference"] = table["Univariate coefficient"] - table["Multivariate coefficient"]
    return table


def trend_table(pvalues: pd.DataFrame, univariate: pd.DataFrame,
                configuration: str = "Untransformed") -> pd.DataFrame:
    """Place each feature's multivariate p-value next to how well its univariate model trained.

    Supports the question "are important predictors also easy to train on their own?".
    """
    if configuration not in pvalues.columns:
        raise ValueError(f"p-value table has no {configuration!r} column")
    significance = pvalues[configuration].rename(f"p-value ({configuration} multivariate)")
    return univariate[["Training R²", "Converged at"]].join(significance, how="left")


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def format_pvalue(p: float) -> str:
    """Three significant digits; values that underflowed to (almost) zero show as "<1e-300"."""
    return f"<{_MIN_PVALUE:.0e}" if p < _MIN_PVALUE else format(p, ".3g")


def to_markdown(table: pd.DataFrame,
                float_format: FloatFormat | Mapping[str, FloatFormat] = ".4g") -> str:
    """Render ``table`` (index included) as a GitHub-flavoured Markdown table.

    Parameters
    ----------
    float_format : format spec, function, or mapping of either
        How floats are shown: one format for all columns, or one per column
        name (unlisted columns use ``".4g"``). Missing values show as "—".
    """
    def cell(column: Any, value: Any) -> str:
        if _is_missing(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            fmt = (float_format.get(column, ".4g") if isinstance(float_format, Mapping)
                   else float_format)
            return fmt(float(value)) if callable(fmt) else format(value, fmt)
        return str(value)

    header = [table.index.name or "", *(str(column) for column in table.columns)]
    align = [":--"] + ["--:" if pd.api.types.is_numeric_dtype(table[column]) else ":--"
                       for column in table.columns]
    lines = [_markdown_row(header), _markdown_row(align)]
    for index, row in table.iterrows():
        lines.append(_markdown_row([str(index), *(cell(c, row[c]) for c in table.columns)]))
    return "\n".join(lines) + "\n"


def save_table(table: pd.DataFrame, name: str, out_dir: PathLike = config.TABLES_DIR,
               float_format: FloatFormat | Mapping[str, FloatFormat] = ".4g") -> list[Path]:
    """Write ``table`` to ``<out_dir>/<name>.csv`` (full precision) and ``<name>.md``.

    The CSV is UTF-8 with a byte-order mark so that Excel shows "R²" correctly.
    """
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path = directory / f"{name}.csv"
    markdown_path = directory / f"{name}.md"
    table.to_csv(csv_path, encoding="utf-8-sig")
    markdown_path.write_text(to_markdown(table, float_format), encoding="utf-8")
    return [csv_path, markdown_path]


def plot_pvalues(pvalues: pd.DataFrame, path: PathLike,
                 level: float = config.SIGNIFICANCE_LEVEL,
                 include_intercept: bool = False) -> Path:
    """Grouped horizontal bar chart of ``-log10(p)`` for every predictor and configuration.

    Longer bars mean stronger evidence that the coefficient is non-zero; the
    dashed line marks the significance level. The x-axis is linear up to 2
    (around the threshold) and logarithmic beyond, so p-values from 0.5 to
    1e-190 stay readable on one chart. Terms keep the table's order.

    Returns
    -------
    Path
        The image written (format from the file extension, e.g. ``.png``).
    """
    from matplotlib.figure import Figure  # imported lazily: only this function needs matplotlib

    if not include_intercept:
        pvalues = pvalues.drop(index="Intercept", errors="ignore")
    scores = -np.log10(pvalues.astype(float).clip(lower=_MIN_PVALUE))
    terms, configurations = list(scores.index), list(scores.columns)
    n_terms, n_configs = len(terms), len(configurations)
    bar_height = 0.8 / max(n_configs, 1)
    positions = np.arange(n_terms)

    figure = Figure(figsize=(7.5, 1.6 + 0.42 * n_terms), facecolor=_SURFACE, layout="constrained")
    axes = figure.subplots()
    axes.set_facecolor(_SURFACE)
    extra_colors = iter(_EXTRA_COLORS)
    for i, configuration in enumerate(configurations):
        color = CONFIG_COLORS.get(configuration) or next(extra_colors)
        offset = (i - (n_configs - 1) / 2) * bar_height
        axes.barh(positions + offset, scores[configuration].to_numpy(), height=bar_height,
                  color=color, edgecolor=_SURFACE, linewidth=0.8, label=configuration, zorder=2)

    threshold = -math.log10(level)
    axes.axvline(threshold, color=_INK_SECONDARY, linestyle=(0, (4, 3)), linewidth=1, zorder=3)
    axes.annotate(f"p = {level:g}", xy=(threshold, 1), xycoords=("data", "axes fraction"),
                  xytext=(4, -2), textcoords="offset points", va="top",
                  color=_INK_SECONDARY, fontsize=8)

    axes.set_yticks(positions, labels=terms)
    axes.invert_yaxis()
    axes.set_xscale("symlog", linthresh=2, linscale=0.6)
    axes.set_xlim(left=0, right=_nice_upper_bound(np.nanmax(scores.to_numpy())))
    axes.xaxis.set_major_formatter("{x:g}")
    axes.set_xlabel("−log10(p-value), symmetric-log scale   (longer bar = stronger evidence)",
                    color=_INK_SECONDARY, fontsize=9)
    axes.set_title("Predictor significance by feature transformation", loc="left",
                   color=_INK, fontsize=12, pad=26)
    axes.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncols=n_configs, frameon=False,
                fontsize=9, handlelength=1.2, borderaxespad=0.2, labelcolor=_INK_SECONDARY)
    axes.grid(axis="x", color=_GRIDLINE, linewidth=0.8, zorder=0)
    axes.tick_params(length=0, labelcolor=_INK_SECONDARY, labelsize=9)
    for side in ("top", "right", "bottom"):
        axes.spines[side].set_visible(False)
    axes.spines["left"].set_color(_BASELINE)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=200, facecolor=_SURFACE)
    return path


# --------------------------------------------------------------------------- #
# End-to-end
# --------------------------------------------------------------------------- #
def build_report(results_dir: PathLike = config.RESULTS_DIR,
                 tables_dir: PathLike = config.TABLES_DIR,
                 figures_dir: PathLike = config.FIGURES_DIR,
                 *, strict: bool = True) -> list[Path]:
    """Write every table and figure the available results allow.

    Tables whose inputs are missing are skipped with a warning, so the report
    can be rebuilt at any stage of the project.

    Returns
    -------
    list of Path
        Every file written.
    """
    if not Path(results_dir).is_dir():
        logger.warning("No results directory at %s; nothing to aggregate.", results_dir)
        return []
    results = collect(results_dir, strict=strict)
    if results.empty:
        logger.warning("No result files in %s; nothing to aggregate.", results_dir)
        return []

    written = save_table(overview_table(results), "all_experiments", tables_dir, _NUMBER_FORMATS)

    pvalues = None
    missing = [exp_id for exp_id in DEFAULT_OLS_IDS if exp_id not in results.index]
    if missing:
        logger.warning("Skipping the optimal-model and p-value tables; missing results: %s "
                       "(run `python -m experiments.c_inference`).", ", ".join(missing))
    else:
        written += save_table(summary_table(results), "optimal_models_summary",
                              tables_dir, _NUMBER_FORMATS)
        pvalues = pvalue_table(results)
        written += save_table(pvalues, "pvalues_by_configuration", tables_dir, format_pvalue)
        written.append(plot_pvalues(pvalues, Path(figures_dir) / "pvalues_by_configuration.png"))

    if DEFAULT_MULTIVARIATE_EXP_ID not in results.index:
        logger.info("No optimal multivariate result (%s); skipping the coefficient "
                    "comparison table.", DEFAULT_MULTIVARIATE_EXP_ID)
    else:
        try:
            comparison = coefficient_comparison_table(results)
        except (MissingResultError, ValueError) as exc:
            logger.warning("Skipping the coefficient comparison table: %s", exc)
        else:
            written += save_table(comparison, "univariate_vs_multivariate_coefficients",
                                  tables_dir, ".6g")

    univariate = univariate_table(results)
    if univariate.empty:
        logger.info("No univariate gradient-descent results yet; skipping the univariate tables.")
    else:
        written += save_table(univariate, "univariate_gd_training", tables_dir, _NUMBER_FORMATS)
        if pvalues is not None:
            formats = {**_NUMBER_FORMATS, "p-value (Untransformed multivariate)": format_pvalue}
            written += save_table(trend_table(pvalues, univariate),
                                  "significance_vs_univariate_training", tables_dir, formats)

    for path in written:
        logger.info("wrote %s", _display_path(path))
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(
        description="Build report tables and figures from result files.")
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--tables-dir", type=Path, default=config.TABLES_DIR)
    parser.add_argument("--figures-dir", type=Path, default=config.FIGURES_DIR)
    parser.add_argument("--lenient", action="store_true",
                        help="skip invalid result files instead of stopping")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        build_report(args.results_dir, args.tables_dir, args.figures_dir, strict=not args.lenient)
    except (ResultSchemaError, MissingResultError, ValueError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


# --------------------------------------------------------------------------- #
# Internal utilities
# --------------------------------------------------------------------------- #
def _select(results: pd.DataFrame, exp_ids: Sequence[str]) -> pd.DataFrame:
    missing = [exp_id for exp_id in exp_ids if exp_id not in results.index]
    if missing:
        raise MissingResultError(
            f"no result file for {', '.join(missing)}; run the experiment(s) that produce "
            "them first (see run_all.py)"
        )
    return results.loc[list(exp_ids)]


def _unique_labels(rows: pd.DataFrame) -> list[str]:
    labels = list(rows["config"])
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(
            f"selected experiments share a configuration ({', '.join(duplicates)}); "
            "select one experiment per configuration"
        )
    return labels


def _matches_multivariate(row: pd.Series, multivariate: pd.Series) -> bool:
    """True for a single-feature result fitted like ``multivariate``: same config and solver."""
    features = row["features"]
    same_solver = (row["solver"] == multivariate["solver"]
                   or (_is_missing(row["solver"]) and _is_missing(multivariate["solver"])))
    return (isinstance(features, list) and len(features) == 1
            and row["config"] == multivariate["config"] and same_solver)


def _is_univariate_gd(row: pd.Series) -> bool:
    features, solver = row["features"], row["solver"]
    used_gd = solver == "gd" or (_is_missing(solver) and not _is_missing(row["alpha"]))
    return used_gd and isinstance(features, list) and len(features) == 1


def _is_missing(value: Any) -> bool:
    if value is None or value is pd.NA:
        return True
    return isinstance(value, (float, np.floating)) and math.isnan(value)


def _markdown_row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cell.replace("|", r"\|") for cell in cells) + " |"


def _nice_upper_bound(value: float) -> float:
    """Smallest of 1, 2, 5 x 10^k that is >= ``value`` (at least 2), for axis limits."""
    value = max(float(value), 2.0)
    magnitude = 10 ** math.floor(math.log10(value))
    return next(step * magnitude for step in (1, 2, 5, 10) if step * magnitude >= value)


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(config.REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    sys.exit(main())
