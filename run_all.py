from __future__ import annotations

import argparse
import enum
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import aggregate
import config
from result_schema import OWNERS

logger = logging.getLogger("run_all")


@dataclass(frozen=True)
class Experiment:
    owner: str
    module: str
    description: str

    @property
    def path(self) -> Path:
        return config.REPO_ROOT.joinpath(*self.module.split(".")).with_suffix(".py")


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment("A", "experiments.a_scaling",
               "Scaling & convergence speed; scaling & the optimal solution"),
    Experiment("B", "experiments.b_log_transform", "Non-linear (log1p) transformation"),
    Experiment("B", "experiments.b_univariate", "Univariate vs. multivariate coefficients"),
    Experiment("C", "experiments.c_inference", "OLS p-values and optimal-model metrics"),
)


class Status(enum.Enum):
    PASSED = "passed"
    FAILED = "FAILED"
    MISSING = "missing"


@dataclass(frozen=True)
class Outcome:
    experiment: Experiment
    status: Status
    seconds: float | None = None


def run_experiment(experiment: Experiment) -> Outcome:
    if not experiment.path.is_file():
        logger.warning("%s: script not found (%s); skipped.",
                       experiment.module, experiment.path.relative_to(config.REPO_ROOT))
        return Outcome(experiment, Status.MISSING)

    logger.info("=== %s (%s): %s", experiment.module, experiment.owner, experiment.description)
    env = {**os.environ, "MPLBACKEND": "Agg"}  # save plots to files, don't open windows
    start = time.perf_counter()
    completed = subprocess.run([sys.executable, "-m", experiment.module],
                               cwd=config.REPO_ROOT, env=env, check=False)
    seconds = time.perf_counter() - start
    if completed.returncode != 0:
        logger.error("%s exited with code %d.", experiment.module, completed.returncode)
        return Outcome(experiment, Status.FAILED, seconds)
    return Outcome(experiment, Status.PASSED, seconds)


def format_summary(outcomes: Sequence[Outcome]) -> str:
    width = max(len(outcome.experiment.module) for outcome in outcomes)
    lines = [f"{'Experiment':<{width}}  Owner  Status   Time"]
    for outcome in outcomes:
        seconds = "-" if outcome.seconds is None else f"{outcome.seconds:.1f}s"
        lines.append(f"{outcome.experiment.module:<{width}}  {outcome.experiment.owner:<5}  "
                     f"{outcome.status.value:<7}  {seconds:>5}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run all experiments, then build the report tables.")
    parser.add_argument("--only", nargs="+", choices=OWNERS, metavar="OWNER",
                        help="run only the experiments of these members (A, B, C)")
    parser.add_argument("--aggregate-only", action="store_true",
                        help="skip the experiments and rebuild tables/figures from results/")
    parser.add_argument("--list", action="store_true", help="list registered experiments and exit")
    parser.add_argument("--strict", action="store_true",
                        help="fail if a registered experiment script is missing")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    selected = [e for e in EXPERIMENTS if not args.only or e.owner in args.only]
    if args.list:
        for experiment in selected:
            state = "ok     " if experiment.path.is_file() else "missing"
            print(f"[{state}] {experiment.owner}  {experiment.module:<28} {experiment.description}")
        return 0

    exit_code = 0
    if not args.aggregate_only:
        try:
            config.require_data_file()
        except FileNotFoundError as exc:
            logger.error("%s", exc)
            logger.error("No experiment can run without the data, so nothing was run. "
                         "Use --aggregate-only to rebuild tables from existing results.")
            return 2
        outcomes = [run_experiment(experiment) for experiment in selected]
        print("\n" + format_summary(outcomes) + "\n", flush=True)
        if any(o.status is Status.FAILED for o in outcomes):
            exit_code = 1
        if args.strict and any(o.status is Status.MISSING for o in outcomes):
            logger.error("--strict: some registered experiment scripts are missing.")
            exit_code = 1

    logger.info("=== Aggregating results")
    try:
        aggregate.build_report()
    except Exception:
        logger.exception("Aggregation failed.")
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
