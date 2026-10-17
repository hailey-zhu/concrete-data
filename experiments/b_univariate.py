from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LinearRegression

import config
import data
from gd import gradient_descent
from metrics import evaluate_split
from result_schema import fields_for_mode, make_result, write_result


logger = logging.getLogger("experiments.b_univariate")

OWNER = "B"
MODE = "raw"


UNIVARIATE_ALPHA_GRID = (
    1e-8,
    3e-8,
    1e-7,
    3e-7,
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
)


def load_raw_data(data_path: Path = config.DATA_PATH):
    X, y, feature_names = data.load_data(data_path)
    X_train, X_test, y_train, y_test = data.split(X, y)

    X_train_raw, params = data.transform(
        X_train,
        mode=MODE,
    )

    X_test_raw, _ = data.transform(
        X_test,
        mode=MODE,
        params=params,
    )

    return (
        np.asarray(X_train_raw, dtype=float),
        np.asarray(X_test_raw, dtype=float),
        np.asarray(y_train, dtype=float),
        np.asarray(y_test, dtype=float),
        tuple(str(name) for name in feature_names),
    )


def run_one_gd(
    X_train,
    X_test,
    y_train,
    y_test,
    alpha: float,
) -> tuple[dict[str, Any], dict[str, float]]:

    gd_result = gradient_descent(
        X_train,
        y_train,
        alpha=alpha,
        n_iters=config.N_ITERS,
        tol=config.TOL,
    )

    metrics = evaluate_split(
        X_train,
        y_train,
        X_test,
        y_test,
        gd_result["theta"],
    )

    return gd_result, metrics


def choose_best_gd_run(
    X_train,
    X_test,
    y_train,
    y_test,
):

    candidates = []

    for alpha in UNIVARIATE_ALPHA_GRID:
        gd_result, metrics = run_one_gd(
            X_train,
            X_test,
            y_train,
            y_test,
            alpha,
        )

        stable = (
            np.isfinite(metrics["train_mse"])
            and np.isfinite(metrics["train_r2"])
            and np.all(np.isfinite(gd_result["theta"]))
        )

        logger.info(
            "    alpha=%-8g train R2=% .4f test R2=% .4f%s",
            alpha,
            metrics["train_r2"],
            metrics["test_r2"],
            "" if stable else "  [diverged]",
        )

        if stable:
            candidates.append(
                {
                    "alpha": alpha,
                    "gd_result": gd_result,
                    "metrics": metrics,
                }
            )

    if not candidates:
        raise RuntimeError(
            "Every learning rate diverged for this feature."
        )

    return min(
        candidates,
        key=lambda run: run["metrics"]["train_mse"],
    )


def run_gradient_descent_models(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_names,
    results_dir: Path,
):

    summaries = []

    for j, feature_name in enumerate(feature_names):
        logger.info(
            "Gradient descent for feature %d/%d: %s",
            j + 1,
            len(feature_names),
            feature_name,
        )

        # Keep X two-dimensional: shape (n_samples, 1).
        X_train_j = X_train[:, [j]]
        X_test_j = X_test[:, [j]]

        best = choose_best_gd_run(
            X_train_j,
            X_test_j,
            y_train,
            y_test,
        )

        alpha = best["alpha"]
        gd_result = best["gd_result"]
        metrics = best["metrics"]

        record = make_result(
            exp_id=f"b_gd_univariate_f{j + 1}",
            owner=OWNER,
            solver="gd",
            **fields_for_mode(MODE),
            alpha=alpha,
            n_iters=config.N_ITERS,
            converged_iter=gd_result["converged_iter"],
            theta=gd_result["theta"],
            features=[feature_name],
            **metrics,
        )

        path = write_result(
            record,
            results_dir,
        )

        logger.info(
            "  best alpha=%g, train R2=%.4f, test R2=%.4f -> %s",
            alpha,
            record["train_r2"],
            record["test_r2"],
            path.name,
        )

        summaries.append(
            {
                "feature_index": j,
                "feature": feature_name,
                "alpha": alpha,
                "train_r2": record["train_r2"],
                "test_r2": record["test_r2"],
                "converged_iter": record["converged_iter"],
            }
        )

    return summaries


def fit_sklearn_univariate_models(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_names,
    results_dir: Path,
):

    records = []

    for j, feature_name in enumerate(feature_names):
        X_train_j = X_train[:, [j]]
        X_test_j = X_test[:, [j]]

        model = LinearRegression()
        model.fit(
            X_train_j,
            y_train,
        )

        theta = np.array(
            [
                float(model.intercept_),
                float(model.coef_[0]),
            ]
        )

        metrics = evaluate_split(
            X_train_j,
            y_train,
            X_test_j,
            y_test,
            theta,
        )

        record = make_result(
            exp_id=f"b_sklearn_univariate_f{j + 1}",
            owner=OWNER,
            solver="sklearn",
            **fields_for_mode(MODE),
            theta=theta,
            features=[feature_name],
            **metrics,
        )

        path = write_result(
            record,
            results_dir,
        )

        logger.info(
            "OLS univariate %-2d slope=% .6f train R2=% .4f -> %s",
            j + 1,
            theta[1],
            record["train_r2"],
            path.name,
        )

        records.append(record)

    return records


def fit_sklearn_multivariate_model(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_names,
    results_dir: Path,
):

    model = LinearRegression()
    model.fit(
        X_train,
        y_train,
    )

    theta = np.concatenate(
        (
            [float(model.intercept_)],
            np.asarray(model.coef_, dtype=float),
        )
    )

    metrics = evaluate_split(
        X_train,
        y_train,
        X_test,
        y_test,
        theta,
    )

    record = make_result(
        exp_id="b_sklearn_multivariate_raw",
        owner=OWNER,
        solver="sklearn",
        **fields_for_mode(MODE),
        theta=theta,
        features=feature_names,
        **metrics,
    )

    path = write_result(
        record,
        results_dir,
    )

    logger.info(
        "Raw multivariate optimum: train R2=%.4f test R2=%.4f -> %s",
        record["train_r2"],
        record["test_r2"],
        path.name,
    )

    return record


def print_coefficient_comparison(
    univariate_records,
    multivariate_record,
):

    print()
    print("Univariate vs. multivariate coefficient comparison")
    print("-" * 105)
    print(
        f"{'#':<3} "
        f"{'Feature':<58} "
        f"{'Univariate':>18} "
        f"{'Multivariate':>18}"
    )
    print("-" * 105)

    for j, univariate in enumerate(univariate_records):
        feature_name = univariate["features"][0]
        uni_coef = univariate["theta"][1]
        multi_coef = multivariate_record["theta"][j + 1]

        # Keep terminal output readable even though UCI column names are long.
        display_name = feature_name[:56]

        print(
            f"{j + 1:<3} "
            f"{display_name:<58} "
            f"{uni_coef:>18.6f} "
            f"{multi_coef:>18.6f}"
        )

    print("-" * 105)


def print_gd_summary(summaries):

    print()
    print("Gradient-descent summary")
    print("-" * 100)
    print(
        f"{'#':<3} "
        f"{'Feature':<48} "
        f"{'Best alpha':>12} "
        f"{'Train R2':>12} "
        f"{'Test R2':>12}"
    )
    print("-" * 100)

    for item in summaries:
        name = item["feature"][:46]

        print(
            f"{item['feature_index'] + 1:<3} "
            f"{name:<48} "
            f"{item['alpha']:>12g} "
            f"{item['train_r2']:>12.4f} "
            f"{item['test_r2']:>12.4f}"
        )

    print("-" * 100)


def main(argv: Sequence[str] | None = None) -> int:

    parser = argparse.ArgumentParser(
        description=(
            "Train eight raw univariate models and compare their "
            "coefficients with the optimal raw multivariate model."
        )
    )

    parser.add_argument(
        "--data-path",
        type=Path,
        default=config.DATA_PATH,
    )

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=config.RESULTS_DIR,
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    (
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
    ) = load_raw_data(args.data_path)

    logger.info(
        "Raw data: %d training rows, %d test rows, %d predictors",
        X_train.shape[0],
        X_test.shape[0],
        X_train.shape[1],
    )

    gd_summaries = run_gradient_descent_models(
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
        args.results_dir,
    )

    univariate_records = fit_sklearn_univariate_models(
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
        args.results_dir,
    )

    multivariate_record = fit_sklearn_multivariate_model(
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
        args.results_dir,
    )

    print_gd_summary(
        gd_summaries,
    )

    print_coefficient_comparison(
        univariate_records,
        multivariate_record,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())