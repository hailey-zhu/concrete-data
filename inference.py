from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats
from statsmodels.regression.linear_model import OLS

from metrics import add_intercept, check_inputs

__all__ = ["NonOptimalThetaWarning", "fit_ols_statsmodels", "ols_inference"]

OPTIMALITY_RTOL: float = 1e-6


class NonOptimalThetaWarning(UserWarning):
    pass


def ols_inference(X: ArrayLike, y: ArrayLike, theta: ArrayLike) -> dict[str, np.ndarray]:
    X, y, theta = check_inputs(X, y, theta)
    if not np.isfinite(theta).all():
        raise ValueError("theta contains NaN or infinite values (did gradient descent diverge?)")
    design = add_intercept(X)
    n_samples, n_params = design.shape
    _check_identifiable(design)

    design_pinv = np.linalg.pinv(design)
    residuals = y - design @ theta
    sse = float(residuals @ residuals)
    _warn_if_not_optimal(design, y, design_pinv, sse)
    if sse == 0.0:
        raise ValueError("residual variance is zero (exact fit); standard errors are undefined")

    # Cov = s^2 (X'X)^-1, se = sqrt(diag(Cov)), t = theta / se, two-sided p-value
    dof = n_samples - n_params
    covariance = (sse / dof) * (design_pinv @ design_pinv.T)
    se = np.sqrt(np.diag(covariance))
    t_stat = theta / se
    p_value = 2.0 * stats.t.sf(np.abs(t_stat), dof)
    return {"se": se, "t_stat": t_stat, "p_value": p_value}


def fit_ols_statsmodels(X: ArrayLike, y: ArrayLike) -> dict[str, np.ndarray]:
    X, y, _ = check_inputs(X, y)
    design = add_intercept(X)
    _check_identifiable(design)
    fitted = OLS(y, design).fit()
    return {
        "theta": np.asarray(fitted.params, dtype=float),
        "se": np.asarray(fitted.bse, dtype=float),
        "t_stat": np.asarray(fitted.tvalues, dtype=float),
        "p_value": np.asarray(fitted.pvalues, dtype=float),
    }


def _check_identifiable(design: np.ndarray) -> None:
    n_samples, n_params = design.shape
    if n_samples <= n_params:
        raise ValueError(
            f"inference needs more samples than parameters, got n={n_samples}, k={n_params}"
        )
    rank = np.linalg.matrix_rank(design)
    if rank < n_params:
        raise ValueError(
            f"design matrix is rank-deficient (rank {rank} < {n_params} parameters): a feature "
            "is constant or perfectly collinear with others, so its coefficient is not identifiable"
        )


def _warn_if_not_optimal(design: np.ndarray, y: np.ndarray,
                         design_pinv: np.ndarray, sse: float) -> None:
    optimal_residuals = y - design @ (design_pinv @ y)
    optimal_sse = float(optimal_residuals @ optimal_residuals)
    excess = (sse - optimal_sse) / max(optimal_sse, np.finfo(float).tiny)
    if excess > OPTIMALITY_RTOL:
        warnings.warn(
            f"theta is not the least-squares solution (its residual sum of squares exceeds "
            f"the optimum by a relative {excess:.2e}), so the reported p-values are approximate",
            NonOptimalThetaWarning,
            stacklevel=3,
        )
