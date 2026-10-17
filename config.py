from __future__ import annotations

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent
DATA_PATH: Path = REPO_ROOT / "data" / "Concrete_Data.csv"
RESULTS_DIR: Path = REPO_ROOT / "results"
FIGURES_DIR: Path = REPO_ROOT / "figures"
TABLES_DIR: Path = REPO_ROOT / "report" / "tables"

TEST_ROWS: slice = slice(500, 630)  # rows 501-630 of the data file
N_TRAIN: int = 900
N_TEST: int = 130

SEED: int = 42
ALPHA_GRID: tuple[float, ...] = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5)
N_ITERS: int = 5000
TOL: float = 1e-8

RAW_ALPHA_GRID: tuple[float, ...] = (1e-7, 3e-7, 5e-7, 7e-7)
RAW_N_ITERS: int = 200_000
RAW_TOL: float = 1e-10

TRANSFORM_MODES: tuple[str, ...] = ("raw", "standardize", "log1p")

OLS_EXP_IDS: dict[str, str] = {mode: f"c_ols_{mode}" for mode in TRANSFORM_MODES}

SIGNIFICANCE_LEVEL: float = 0.05


def require_data_file(path: str | Path | None = None) -> Path:
    path = DATA_PATH if path is None else Path(path)
    if path.is_file():
        return path
    found = sorted({candidate for pattern in ("Concrete_Data.*", "*/Concrete_Data.*")
                    for candidate in REPO_ROOT.glob(pattern)})
    if found:
        hint = "the repository contains " + ", ".join(_display(p) for p in found)
    else:
        hint = "no Concrete_Data.* file was found in the repository"
    raise FileNotFoundError(
        f"data file not found: {_display(path)} ({hint}). "
        "Point config.DATA_PATH at the data file, or pass --data-path."
    )


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)
