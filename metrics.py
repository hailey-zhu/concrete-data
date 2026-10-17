from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "add_intercept", "check_inputs", "evaluate", "evaluate_split",
    "mse", "predict", "r_squared",
]


def check_inputs(
    X: ArrayLike, y: ArrayLike, theta: ArrayLike | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    X = _as_feature_matrix(X)
    y = np.asarray(y, dtype=float)
    if y.ndim == 2 and y.shape[1] == 1:
        y = y[:, 0]
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D with shape (n_samples,), got shape {y.shape}")
    if y.shape[0] != X.shape[0]:
        raise ValueError(f"X has {X.shape[0]} samples but y has {y.shape[0]}")
    if not np.isfinite(y).all():
        raise ValueError("y contains NaN or infinite values")
    if theta is not None:
        theta = _as_parameter_vector(theta, X.shape[1])
    return X, y, theta


def add_intercept(X: ArrayLike) -> np.ndarray:
    X = _as_feature_matrix(X)
    return np.column_stack([np.ones(X.shape[0]), X])


def predict(X: ArrayLike, theta: ArrayLike) -> np.ndarray:
    X = _as_feature_matrix(X)
    theta = _as_parameter_vector(theta, X.shape[1])
    with np.errstate(over="ignore", invalid="ignore"):
        return theta[0] + X @ theta[1:]


def mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true and y_pred must be 1-D with equal length, got shapes "
            f"{y_true.shape} and {y_pred.shape}"
        )
    if y_true.size == 0:
        raise ValueError("cannot compute MSE of zero samples")
    with np.errstate(over="ignore", invalid="ignore"):
        return float(np.mean((y_true - y_pred) ** 2))


def r_squared(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    # R^2 = 1 - MSE / Var(y)
    error = mse(y_true, y_pred)
    variance = float(np.var(np.asarray(y_true, dtype=float)))
    if variance == 0.0:
        raise ValueError("R² is undefined when the observed values are constant (zero variance)")
    return 1.0 - error / variance


def evaluate(X: ArrayLike, y: ArrayLike, theta: ArrayLike) -> dict[str, float]:
    X, y, theta = check_inputs(X, y, theta)
    y_pred = predict(X, theta)
    return {"mse": mse(y, y_pred), "r2": r_squared(y, y_pred)}


def evaluate_split(
    X_train: ArrayLike, y_train: ArrayLike,
    X_test: ArrayLike, y_test: ArrayLike,
    theta: ArrayLike,
) -> dict[str, float]:
    train = evaluate(X_train, y_train, theta)
    test = evaluate(X_test, y_test, theta)
    return {
        "train_mse": train["mse"],
        "test_mse": test["mse"],
        "train_r2": train["r2"],
        "test_r2": test["r2"],
    }


def _as_feature_matrix(X: ArrayLike) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"X must be 1-D or 2-D, got an array with shape {X.shape}")
    if X.shape[0] == 0:
        raise ValueError("X must contain at least one sample")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values")
    return X


def _as_parameter_vector(theta: ArrayLike, n_features: int) -> np.ndarray:
    theta = np.asarray(theta, dtype=float)
    expected = n_features + 1
    if theta.shape != (expected,):
        raise ValueError(
            f"theta must have shape ({expected},), i.e. n_features + 1 with the "
            f"intercept first, got shape {theta.shape}"
        )
    return theta
