from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import TEST_ROWS


def load_data(path):
    path = Path(path)

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xls", ".xlsx"}:
        df = pd.read_excel(path)
    else:
        raise ValueError(
            f"Unsupported file type: {path.suffix}. "
            "Expected .csv, .xls, or .xlsx."
        )

    # The UCI concrete dataset contains 8 predictors followed by the target.
    if df.shape[1] != 9:
        raise ValueError(
            f"Expected 9 columns (8 predictors + target), got {df.shape[1]}."
        )

    feature_names = list(df.columns[:-1])

    X = df.iloc[:, :-1].to_numpy(dtype=float)
    y = df.iloc[:, -1].to_numpy(dtype=float)

    return X, y, feature_names


def split(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1)

    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")

    test_mask = np.zeros(len(X), dtype=bool)
    test_mask[TEST_ROWS] = True

    X_test = X[test_mask]
    y_test = y[test_mask]

    X_train = X[~test_mask]
    y_train = y[~test_mask]

    return X_train, X_test, y_train, y_test


def transform(X, mode, params=None):
    X = np.asarray(X, dtype=float)

    was_1d = X.ndim == 1

    if was_1d:
        X = X.reshape(-1, 1)

    if mode == "raw":
        X_out = X.copy()

        out_params = {
            "mode": "raw",
        }

    elif mode == "standardize":
        if params is None:
            mean = np.mean(X, axis=0)
            std = np.std(X, axis=0)

            # Defensive check in case a feature is constant.
            if np.any(std == 0):
                raise ValueError(
                    "Cannot standardize a feature with zero variance."
                )

            out_params = {
                "mode": "standardize",
                "mean": mean,
                "std": std,
            }

        else:
            mean = np.asarray(params["mean"], dtype=float)
            std = np.asarray(params["std"], dtype=float)

            out_params = params

        X_out = (X - mean) / std

    elif mode == "log1p":
        # log(1 + x) is only valid here when x > -1.
        if np.any(X <= -1):
            raise ValueError(
                "log1p transformation requires all feature values to be > -1."
            )

        X_out = np.log1p(X)

        out_params = {
            "mode": "log1p",
        }

    else:
        raise ValueError(
            f"Unknown transform mode: {mode}. "
            "Expected 'raw', 'standardize', or 'log1p'."
        )

    if was_1d:
        X_out = X_out.reshape(-1)

    return X_out, out_params


def recover_coef(theta, params):
    theta = np.asarray(theta, dtype=float).reshape(-1)

    mode = params["mode"]

    if mode == "raw":
        return theta.copy()

    if mode == "standardize":
        mean = np.asarray(params["mean"], dtype=float)
        std = np.asarray(params["std"], dtype=float)

        intercept_scaled = theta[0]
        weights_scaled = theta[1:]

        if len(weights_scaled) != len(mean):
            raise ValueError(
                "theta and transformation parameters have incompatible sizes."
            )

        weights_original = weights_scaled / std

        intercept_original = (
            intercept_scaled
            - np.sum(weights_scaled * mean / std)
        )

        return np.concatenate(
            ([intercept_original], weights_original)
        )

    if mode == "log1p":
        raise ValueError(
            "Coefficients from a log1p model cannot be converted into "
            "equivalent linear coefficients on the raw feature scale."
        )

    raise ValueError(f"Unknown transform mode: {mode}.")