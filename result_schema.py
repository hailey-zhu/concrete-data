from __future__ import annotations

import json
import math
import os
import re
import reprlib
import tempfile
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import config

__all__ = [
    "ALL_FIELDS", "METRIC_FIELDS", "OPTIONAL_FIELDS", "REQUIRED_FIELDS",
    "ResultSchemaError", "config_label", "fields_for_mode", "make_result",
    "read_result", "validate_result", "write_result",
]

REQUIRED_FIELDS: tuple[str, ...] = (
    "exp_id", "owner", "transform", "scaling", "alpha", "n_iters", "theta",
    "train_mse", "test_mse", "train_r2", "test_r2", "converged_iter", "p_value",
)
OPTIONAL_FIELDS: tuple[str, ...] = ("solver", "features")
ALL_FIELDS: tuple[str, ...] = REQUIRED_FIELDS + OPTIONAL_FIELDS
METRIC_FIELDS: tuple[str, ...] = ("train_mse", "test_mse", "train_r2", "test_r2")

OWNERS: tuple[str, ...] = ("A", "B", "C")
TRANSFORMS: tuple[str, ...] = ("none", "log1p")
SCALINGS: tuple[str, ...] = ("none", "standardize", "normalize")
SOLVERS: tuple[str, ...] = ("gd", "sklearn", "statsmodels", "normal_equation")

# data.transform() mode -> (transform, scaling)
_MODE_FIELDS: dict[str, tuple[str, str]] = {
    "raw": ("none", "none"),
    "standardize": ("none", "standardize"),
    "normalize": ("none", "normalize"),
    "log1p": ("log1p", "none"),
}
_CONFIG_LABELS: dict[tuple[str, str], str] = {
    ("none", "none"): "Untransformed",
    ("none", "standardize"): "Standardized",
    ("none", "normalize"): "Normalized",
    ("log1p", "none"): "Log-transformed",
}
_EXP_ID_PATTERN = re.compile(r"[a-z0-9_]+")
_R2_UPPER_BOUND = 1.0 + 1e-9

_short_repr = reprlib.Repr()
_short_repr.maxlist = 4
_short_repr.maxstring = 60


class ResultSchemaError(ValueError):
    pass


def fields_for_mode(mode: str) -> dict[str, str]:
    try:
        transform, scaling = _MODE_FIELDS[mode]
    except KeyError:
        raise ValueError(
            f"unknown transform mode {mode!r}; expected one of {', '.join(_MODE_FIELDS)}"
        ) from None
    return {"transform": transform, "scaling": scaling}


def config_label(transform: str, scaling: str) -> str:
    return _CONFIG_LABELS.get((transform, scaling), f"{transform} + {scaling}")


def make_result(
    *,
    exp_id: str,
    owner: str,
    transform: str,
    scaling: str,
    theta: Sequence[float] | np.ndarray,
    train_mse: float,
    test_mse: float,
    train_r2: float,
    test_r2: float,
    alpha: float | None = None,
    n_iters: int | None = None,
    converged_iter: int | None = None,
    p_value: Sequence[float] | np.ndarray | None = None,
    solver: str | None = None,
    features: Sequence[str] | None = None,
) -> dict[str, Any]:
    record = _to_builtin({
        "exp_id": exp_id,
        "owner": owner,
        "transform": transform,
        "scaling": scaling,
        "alpha": alpha,
        "n_iters": n_iters,
        "theta": theta,
        "train_mse": train_mse,
        "test_mse": test_mse,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "converged_iter": converged_iter,
        "p_value": p_value,
        "solver": solver,
        "features": features,
    })
    validate_result(record)
    return record


def validate_result(record: Mapping[str, Any]) -> None:
    if not isinstance(record, Mapping):
        raise ResultSchemaError(
            f"a result record must be a JSON object, got {type(record).__name__}"
        )

    problems: list[str] = []
    missing = [name for name in REQUIRED_FIELDS if name not in record]
    if missing:
        problems.append(f"missing required field(s): {', '.join(missing)}")
    unknown = sorted(set(record) - set(ALL_FIELDS))
    if unknown:
        problems.append(
            f"unknown field(s): {', '.join(unknown)} "
            "(new fields need group sign-off, then an entry in OPTIONAL_FIELDS)"
        )

    def check(name: str, is_valid: Callable[[Any], bool], expected: str,
              nullable: bool = False) -> None:
        if name not in record or (nullable and record[name] is None):
            return
        if not is_valid(record[name]):
            problems.append(f"{name} must be {expected}, got {_short_repr.repr(record[name])}")

    check("exp_id", lambda v: isinstance(v, str) and _EXP_ID_PATTERN.fullmatch(v) is not None,
          "a non-empty string of lowercase letters, digits and underscores")
    check("owner", lambda v: v in OWNERS, f"one of {', '.join(OWNERS)}")
    check("transform", lambda v: v in TRANSFORMS, f"one of {', '.join(TRANSFORMS)}")
    check("scaling", lambda v: v in SCALINGS, f"one of {', '.join(SCALINGS)}")
    check("solver", lambda v: v in SOLVERS, f"null or one of {', '.join(SOLVERS)}", nullable=True)
    check("alpha", lambda v: _is_real(v) and math.isfinite(v) and v > 0,
          "null or a positive number", nullable=True)
    check("n_iters", lambda v: _is_int(v) and v >= 1, "null or a positive integer", nullable=True)
    check("converged_iter", lambda v: _is_int(v) and v >= 1,
          "null or a positive integer", nullable=True)
    check("theta", lambda v: _is_number_list(v) and len(v) >= 1,
          "a non-empty list of numbers (intercept first)")
    for name in ("train_mse", "test_mse"):
        check(name, lambda v: _is_real(v) and not v < 0,
              "a non-negative number (NaN/Infinity allowed)")
    for name in ("train_r2", "test_r2"):
        check(name, lambda v: _is_real(v) and not v > _R2_UPPER_BOUND,
              "a number no greater than 1 (NaN/-Infinity allowed)")
    check("p_value", lambda v: _is_number_list(v) and all(0.0 <= p <= 1.0 for p in v),
          "null or a list of probabilities in [0, 1]", nullable=True)
    check("features",
          lambda v: isinstance(v, (list, tuple)) and len(set(v)) == len(v)
          and all(isinstance(name, str) and name for name in v),
          "null or a list of unique, non-empty names", nullable=True)

    problems.extend(_cross_field_problems(record))
    if problems:
        label = record.get("exp_id")
        heading = "invalid result record" + (f" {label!r}" if isinstance(label, str) else "")
        raise ResultSchemaError(heading + ":\n  - " + "\n  - ".join(problems))


def _cross_field_problems(record: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    exp_id, owner = record.get("exp_id"), record.get("owner")
    if isinstance(exp_id, str) and owner in OWNERS and not exp_id.startswith(f"{owner.lower()}_"):
        problems.append(
            f"exp_id must start with '{owner.lower()}_' for owner {owner} "
            "(keeps members' file names from colliding)"
        )

    theta = record.get("theta")
    if _is_number_list(theta):
        p_value = record.get("p_value")
        if _is_number_list(p_value) and len(p_value) != len(theta):
            problems.append(f"p_value has {len(p_value)} entries but theta has {len(theta)}")
        features = record.get("features")
        if isinstance(features, (list, tuple)) and len(features) != len(theta) - 1:
            problems.append(
                f"features has {len(features)} names but theta has {len(theta) - 1} weights"
            )

    n_iters, converged_iter = record.get("n_iters"), record.get("converged_iter")
    if _is_int(n_iters) and _is_int(converged_iter) and converged_iter > n_iters:
        problems.append(f"converged_iter ({converged_iter}) exceeds n_iters ({n_iters})")
    return problems


def write_result(record: Mapping[str, Any],
                 results_dir: str | os.PathLike[str] = config.RESULTS_DIR) -> Path:
    record = _to_builtin(dict(record))
    validate_result(record)
    directory = Path(results_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record['exp_id']}.json"
    ordered = {name: record[name] for name in ALL_FIELDS if name in record}
    _atomic_write_text(path, json.dumps(ordered, indent=2) + "\n")
    return path


def read_result(path: str | os.PathLike[str]) -> dict[str, Any]:
    path = Path(path)
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ResultSchemaError(f"{path.name}: not valid JSON ({exc})") from exc
    try:
        validate_result(record)
    except ResultSchemaError as exc:
        raise ResultSchemaError(f"{path.name}: {exc}") from None
    if record["exp_id"] != path.stem:
        raise ResultSchemaError(
            f"{path.name}: exp_id {record['exp_id']!r} does not match the file name"
        )
    for name in OPTIONAL_FIELDS:
        record.setdefault(name, None)
    return record


def _is_real(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _is_number_list(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and all(_is_real(item) for item in value)


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _to_builtin(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value


def _atomic_write_text(path: Path, text: str) -> None:
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
