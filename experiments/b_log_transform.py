from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

import config
import data
from gd import gradient_descent
from metrics import evaluate_split
from result_schema import fields_for_mode, make_result, write_result


logger = logging.getLogger("experiments.b_log_transform")

OWNER = "B"
MODE = "log1p"


def alpha_tag(alpha: float) -> str:
    return f"{alpha:g}".replace(".", "p")


def load_log_data(data_path: Path = config.DATA_PATH):
    X, y, feature_names = data.load_data(data_path)

    X_train, X_test, y_train, y_test = data.split(X, y)

    X_train_log, params = data.transform(
        X_train,
        mode=MODE,
    )

    X_test_log, _ = data.transform(
        X_test,
        mode=MODE,
        params=params,
    )

    return (
        X_train_log,
        X_test_log,
        y_train,
        y_test,
        feature_names,
    )


def run_gradient_descent(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_names,
    results_dir: Path,
):

    runs = []

    for alpha in config.ALPHA_GRID:
        logger.info("Running gradient descent with alpha=%s", alpha)

        gd_result = gradient_descent(
            X_train,
            y_train,
            alpha=alpha,
            n_iters=config.N_ITERS,
            tol=config.TOL,
        )

        theta = gd_result["theta"]

        metrics = evaluate_split(
            X_train,
            y_train,
            X_test,
            y_test,
            theta,
        )

        record = make_result(
            exp_id=f"b_gd_log1p_a{alpha_tag(alpha)}",
            owner=OWNER,
            solver="gd",
            **fields_for_mode(MODE),
            alpha=alpha,
            n_iters=config.N_ITERS,
            converged_iter=gd_result["converged_iter"],
            theta=theta,
            features=feature_names,
            **metrics,
        )

        path = write_result(record, results_dir)

        logger.info(
            "alpha=%-7s train R2=% .4f test R2=% .4f -> %s",
            alpha,
            record["train_r2"],
            record["test_r2"],
            path.name,
        )

        runs.append(
            {
                "alpha": alpha,
                "loss_history": gd_result["loss_history"],
                "record": record,
            }
        )

    return runs


def fit_sklearn_optimum(
    X_train,
    X_test,
    y_train,
    y_test,
    feature_names,
    results_dir: Path,
):

    model = LinearRegression()
    model.fit(X_train, y_train)

    theta = np.concatenate(
        (
            [model.intercept_],
            model.coef_,
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
        exp_id="b_sklearn_log1p",
        owner=OWNER,
        solver="sklearn",
        **fields_for_mode(MODE),
        theta=theta,
        features=feature_names,
        **metrics,
    )

    path = write_result(record, results_dir)

    logger.info(
        "sklearn optimum: train R2=%.4f test R2=%.4f -> %s",
        record["train_r2"],
        record["test_r2"],
        path.name,
    )

    return record


def save_loss_figure(runs, figures_dir: Path):

    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    plotted_any = False

    for run in runs:
        losses = np.asarray(
            run["loss_history"],
            dtype=float,
        )

        # Skip runs that diverged.
        if not np.all(np.isfinite(losses)):
            logger.warning(
                "Skipping divergent alpha=%s in loss figure",
                run["alpha"],
            )
            continue

        iterations = np.arange(
            1,
            len(losses) + 1,
        )

        plt.plot(
            iterations,
            losses,
            label=f"alpha={run['alpha']}",
        )

        plotted_any = True

    if not plotted_any:
        logger.warning(
            "No finite gradient-descent losses were available to plot."
        )
        plt.close()
        return None

    plt.yscale("log")
    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss")
    plt.title("Gradient Descent on Log-Transformed Features")
    plt.legend()
    plt.tight_layout()

    path = figures_dir / "b_log1p_loss.png"

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    logger.info("Saved loss figure -> %s", path)

    return path


def main(argv: Sequence[str] | None = None) -> int:

    parser = argparse.ArgumentParser(
        description="Log1p transformation gradient-descent experiment."
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

    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=config.FIGURES_DIR,
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
    ) = load_log_data(args.data_path)

    logger.info(
        "Log-transformed data: %d training rows, %d test rows, %d features",
        X_train.shape[0],
        X_test.shape[0],
        X_train.shape[1],
    )

    runs = run_gradient_descent(
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
        args.results_dir,
    )

    fit_sklearn_optimum(
        X_train,
        X_test,
        y_train,
        y_test,
        feature_names,
        args.results_dir,
    )

    save_loss_figure(
        runs,
        args.figures_dir,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())