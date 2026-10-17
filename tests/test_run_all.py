"""Tests for run_all.py (registry and reporting; experiments themselves are not executed)."""
import contextlib
import io
import unittest
from pathlib import Path
from unittest import mock

import config
import run_all
from result_schema import OWNERS

MISSING_DATA = Path("no_such_folder") / "Concrete_Data.xls"


class RegistryTests(unittest.TestCase):
    def test_registry_entries_are_well_formed(self):
        modules = [experiment.module for experiment in run_all.EXPERIMENTS]
        self.assertEqual(len(modules), len(set(modules)), "duplicate module in EXPERIMENTS")
        for experiment in run_all.EXPERIMENTS:
            self.assertIn(experiment.owner, OWNERS)
            prefix = f"experiments.{experiment.owner.lower()}_"
            self.assertTrue(experiment.module.startswith(prefix), experiment.module)
            self.assertEqual(experiment.path.suffix, ".py")

    def test_person_c_experiment_is_registered_and_present(self):
        c_experiments = [e for e in run_all.EXPERIMENTS if e.owner == "C"]
        self.assertTrue(c_experiments)
        self.assertTrue(all(e.path.is_file() for e in c_experiments))

    def test_list_option(self):
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exit_code = run_all.main(["--list", "--only", "C"])
        self.assertEqual(exit_code, 0)
        self.assertIn("experiments.c_inference", buffer.getvalue())
        self.assertNotIn("experiments.a_", buffer.getvalue())

    def test_missing_script_is_reported_not_run(self):
        ghost = run_all.Experiment("A", "experiments.a_does_not_exist", "placeholder")
        with self.assertLogs("run_all", level="WARNING"):
            outcome = run_all.run_experiment(ghost)
        self.assertIs(outcome.status, run_all.Status.MISSING)

    def test_missing_data_file_stops_before_running_experiments(self):
        with (mock.patch.object(config, "DATA_PATH", MISSING_DATA),
              mock.patch.object(run_all, "run_experiment") as run_experiment,
              self.assertLogs("run_all", level="ERROR") as logs):
            exit_code = run_all.main([])
        self.assertEqual(exit_code, 2)
        run_experiment.assert_not_called()
        self.assertIn("data file not found", logs.output[0])

    def test_aggregate_only_does_not_need_the_data_file(self):
        with (mock.patch.object(config, "DATA_PATH", MISSING_DATA),
              mock.patch.object(run_all.aggregate, "build_report") as build_report,
              self.assertLogs("run_all", level="INFO")):
            exit_code = run_all.main(["--aggregate-only"])
        self.assertEqual(exit_code, 0)
        build_report.assert_called_once()

    def test_summary_lists_every_outcome(self):
        outcomes = [run_all.Outcome(run_all.EXPERIMENTS[0], run_all.Status.MISSING),
                    run_all.Outcome(run_all.EXPERIMENTS[-1], run_all.Status.PASSED, 2.5)]
        summary = run_all.format_summary(outcomes)
        self.assertIn("missing", summary)
        self.assertIn("2.5s", summary)


if __name__ == "__main__":
    unittest.main()
