from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import config
from inference import fit_ols_statsmodels, ols_inference
from metrics import evaluate_split
from result_schema import fields_for_mode, make_result, write_result

logger = logging.getLogger("experiments.c_inference")

OWNER = "C"
CROSS_CHECK_RTOL = 1e-6


@dataclass(frozen=True, eq=False)
class FeatureSet:
    mode: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]


def load_feature_sets(modes: Sequence[str] = config.TRANSFORM_MODES,
                      data_path: Path = config.DATA_PATH) -> list[FeatureSet]:
    import data

    X, y, feature_names = data.load_data(data_path)
    X_train, X_test, y_train, y_test = data.split(X, y)
    _check_partition(np.asarray(X, dtype=float), np.asarray(X_train, dtype=float),
                     np.asarray(X_test, dtype=float))

    feature_sets = []
    for mode in modes:
        X_train_mode, params = data.transform(X_train, mode=mode)
        X_test_mode, _ = data.transform(X_test, mode=mode, params=params)
        feature_sets.append(FeatureSet(
            mode=mode,
            X_train=np.asarray(X_train_mode, dtype=float),
            X_test=np.asarray(X_test_mode, dtype=float),
            y_train=np.asarray(y_train, dtype=float),
            y_test=np.asarray(y_test, dtype=float),
            feature_names=tuple(str(name) for name in feature_names),
        ))
    return feature_sets


def run_configuration(features: FeatureSet) -> dict[str, Any]:
    ols = fit_ols_statsmodels(features.X_train, features.y_train)
    # check statsmodels against our own formulas in inference.py
    manual = ols_inference(features.X_train, features.y_train, ols["theta"])
    _cross_check(features.mode, manual, ols)
    metrics = evaluate_split(features.X_train, features.y_train,
                             features.X_test, features.y_test, ols["theta"])
    return make_result(
        exp_id=config.OLS_EXP_IDS[features.mode],
        owner=OWNER,
        solver="statsmodels",
        **fields_for_mode(features.mode),
        theta=ols["theta"],
        p_value=ols["p_value"],
        features=features.feature_names,
        **metrics,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OLS p-values and train/test metrics "
                                                 "for the three feature configurations.")
    parser.add_argument("--results-dir", type=Path, default=config.RESULTS_DIR)
    parser.add_argument("--data-path", type=Path, default=config.DATA_PATH)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        data_path = config.require_data_file(args.data_path)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2

    try:
        feature_sets = load_feature_sets(data_path=data_path)
    except ModuleNotFoundError as exc:
        if exc.name != "data":
            raise
        logger.error("data.py (Person B) could not be imported. Run from the repository root "
                     "with `python -m experiments.c_inference` once data.py is in place.")
        return 2

    for features in feature_sets:
        record = run_configuration(features)
        path = write_result(record, args.results_dir)
        logger.info("%-11s train R2 = %.4f, test R2 = %.4f -> %s", features.mode,
                    record["train_r2"], record["test_r2"], path.name)
    return 0


def _check_partition(X: np.ndarray, X_train: np.ndarray, X_test: np.ndarray) -> None:
    def same_rows(actual: np.ndarray, expected: np.ndarray) -> bool:
        return (actual.shape == expected.shape
                and sorted(map(tuple, actual)) == sorted(map(tuple, expected)))

    if len(X) != config.N_TRAIN + config.N_TEST:
        raise RuntimeError(
            f"data.load_data() returned {len(X)} rows, but the assignment's partition refers to "
            f"all {config.N_TRAIN + config.N_TEST} rows of the file (no rows may be dropped "
            "before splitting)"
        )
    expected_test = X[config.TEST_ROWS]
    expected_train = np.delete(X, np.arange(len(X))[config.TEST_ROWS], axis=0)
    if not (same_rows(X_test, expected_test) and same_rows(X_train, expected_train)):
        raise RuntimeError(
            f"data.split() returned {len(X_train)} training / {len(X_test)} test rows that do "
            f"not match the assignment: rows 501-630 must be the {config.N_TEST}-row test set "
            f"and the remaining {config.N_TRAIN} rows the training set"
        )


def _cross_check(mode: str, manual: dict[str, np.ndarray],
                 reference: dict[str, np.ndarray]) -> None:
    for key in ("se", "t_stat", "p_value"):
        if not np.allclose(manual[key], reference[key], rtol=CROSS_CHECK_RTOL, atol=1e-12):
            worst = np.max(np.abs(manual[key] - reference[key]))
            raise RuntimeError(
                f"{mode}: hand-written {key} disagrees with statsmodels "
                f"(max abs difference {worst:.3e})"
            )


if __name__ == "__main__":
    sys.exit(main())
