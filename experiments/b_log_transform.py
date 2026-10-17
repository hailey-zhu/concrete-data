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
    return f"{alpha:g}".replace(".", "p").replace("-", "m")


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

def _training_r2(X, y, theta):
    X_design = np.hstack([np.ones((X.shape[0], 1)), X])
    y_pred = X_design @ theta
    mse = np.mean((y - y_pred) ** 2)
    return 1 - mse / np.var(y)


def save_transform_comparison_figure(
    data_path: Path,
    log_runs,
    figures_dir: Path,
):
    figures_dir.mkdir(parents=True, exist_ok=True)

    X, y, _ = data.load_data(data_path)
    X_train, _, y_train, _ = data.split(X, y)

    # Standardized training features
    X_train_std, _ = data.transform(
        X_train,
        mode="standardize",
    )

    # Match Person A's raw-model selection:
    # choose the stable raw learning rate with the highest positive train R2.
    raw_candidates = []

    for alpha in config.RAW_ALPHA_GRID:
        result = gradient_descent(
            X_train,
            y_train,
            alpha=alpha,
            n_iters=config.RAW_N_ITERS,
            tol=config.RAW_TOL,
        )

        final_loss = result["loss_history"][-1]

        if np.isfinite(final_loss):
            train_r2 = _training_r2(
                X_train,
                y_train,
                result["theta"],
            )

            if train_r2 > 0:
                raw_candidates.append(
                    (alpha, train_r2, result)
                )

    if not raw_candidates:
        raise RuntimeError(
            "No raw-feature gradient-descent run achieved positive train R2."
        )

    raw_alpha, _, res_raw = max(
        raw_candidates,
        key=lambda item: item[1],
    )

    # Match Person A's standardized-model selection:
    # choose the positive-R2 converged run that converges fastest.
    std_candidates = []

    for alpha in config.ALPHA_GRID:
        result = gradient_descent(
            X_train_std,
            y_train,
            alpha=alpha,
            n_iters=config.N_ITERS,
            tol=config.TOL,
        )

        final_loss = result["loss_history"][-1]

        if not np.isfinite(final_loss):
            continue

        train_r2 = _training_r2(
            X_train_std,
            y_train,
            result["theta"],
        )

        if result["converged_iter"] and train_r2 > 0:
            std_candidates.append(
                (
                    alpha,
                    result["converged_iter"],
                    result,
                )
            )

    if not std_candidates:
        raise RuntimeError(
            "No standardized gradient-descent run converged with positive train R2."
        )

    std_alpha, _, res_std = min(
        std_candidates,
        key=lambda item: item[1],
    )

    # Use the best stable log-transformed run already produced above.
    stable_log_runs = [
        run
        for run in log_runs
        if np.isfinite(run["record"]["train_r2"])
        and np.all(np.isfinite(run["loss_history"]))
    ]

    if not stable_log_runs:
        raise RuntimeError(
            "No stable log-transformed gradient-descent run was available."
        )

    best_log_run = max(
        stable_log_runs,
        key=lambda run: run["record"]["train_r2"],
    )

    log_alpha = best_log_run["alpha"]
    res_log_losses = best_log_run["loss_history"]

    # Figure 4
    plt.figure(figsize=(8, 5.5))

    plt.plot(
        res_raw["loss_history"],
        label=f"Raw features (alpha={raw_alpha})",
    )

    plt.plot(
        res_std["loss_history"],
        label=f"Standardized features (alpha={std_alpha})",
    )

    plt.plot(
        res_log_losses,
        label=f"Log-transformed features (alpha={log_alpha})",
    )

    plt.yscale("log")
    plt.xlim(0, config.N_ITERS)

    plt.xlabel("Iteration")
    plt.ylabel("MSE Loss (log scale)")
    plt.title("Gradient Descent Loss by Feature Transformation")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    path = figures_dir / "q5_1_loss_comparison.png"

    plt.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()

    logger.info(
        "Saved transformation comparison figure -> %s "
        "(raw alpha=%s, standardized alpha=%s, log alpha=%s)",
        path,
        raw_alpha,
        std_alpha,
        log_alpha,
    )

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

    save_transform_comparison_figure(
        args.data_path,
        runs,
        args.figures_dir,
    )

    return 0




if __name__ == "__main__":
    sys.exit(main())
    