"""Integration test for experiments/c_inference.py against a fake ``data`` module.

The fake follows the interface agreed with Person B, so this test also pins
down what Person C's code expects from data.py.
"""
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

import config
from experiments import c_inference
from result_schema import read_result

N_ROWS = config.N_TRAIN + config.N_TEST
NAMES = ["Cement", "Water", "Age"]


def _fake_data_module(split_rows=config.TEST_ROWS):
    rng = np.random.default_rng(0)
    X = rng.uniform(1, 500, size=(N_ROWS, len(NAMES)))
    y = 10 + 0.1 * X[:, 0] - 0.05 * X[:, 1] + 3 * np.log1p(X[:, 2]) + rng.normal(size=N_ROWS)

    def load_data(path):
        return X.copy(), y.copy(), list(NAMES)

    def split(X, y):
        test = np.zeros(len(X), dtype=bool)
        test[split_rows] = True
        return X[~test], X[test], y[~test], y[test]

    def transform(X, mode, params=None):
        if mode == "raw":
            return X.copy(), {}
        if mode == "log1p":
            return np.log1p(X), {}
        if mode == "standardize":
            params = params or {"mean": X.mean(axis=0), "std": X.std(axis=0)}
            return (X - params["mean"]) / params["std"], params
        raise ValueError(mode)

    module = types.ModuleType("data")
    module.load_data, module.split, module.transform = load_data, split, transform
    return module


class CInferenceTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.results_dir = Path(self._tmp.name)
        # the fake data module ignores the path, but main() checks that the file exists
        self.data_file = self.results_dir / "Concrete_Data.csv"
        self.data_file.write_text("", encoding="utf-8")

    def tearDown(self):
        self._tmp.cleanup()

    def test_writes_one_valid_result_per_mode(self):
        with mock.patch.dict(sys.modules, {"data": _fake_data_module()}):
            with self.assertLogs("experiments.c_inference", level="INFO"):
                exit_code = c_inference.main(["--results-dir", str(self.results_dir),
                                              "--data-path", str(self.data_file)])
        self.assertEqual(exit_code, 0)
        for mode, exp_id in config.OLS_EXP_IDS.items():
            record = read_result(self.results_dir / f"{exp_id}.json")
            self.assertEqual(record["features"], NAMES)
            self.assertEqual(len(record["p_value"]), len(NAMES) + 1)
            self.assertEqual(record["solver"], "statsmodels")

    def test_standardizing_changes_only_the_intercept_pvalue(self):
        with mock.patch.dict(sys.modules, {"data": _fake_data_module()}):
            records = {fs.mode: c_inference.run_configuration(fs)
                       for fs in c_inference.load_feature_sets()}
        np.testing.assert_allclose(records["standardize"]["p_value"][1:],
                                   records["raw"]["p_value"][1:], rtol=1e-6)
        self.assertAlmostEqual(records["standardize"]["train_r2"], records["raw"]["train_r2"])

    def test_rejects_a_split_that_ignores_the_assignment(self):
        with mock.patch.dict(sys.modules, {"data": _fake_data_module(split_rows=slice(900, None))}):
            with self.assertRaisesRegex(RuntimeError, "rows 501-630"):
                c_inference.load_feature_sets()

    def test_missing_data_file_gives_clear_exit_code(self):
        missing = self.results_dir / "no_such_file.csv"
        with self.assertLogs("experiments.c_inference", level="ERROR") as logs:
            exit_code = c_inference.main(["--results-dir", str(self.results_dir),
                                          "--data-path", str(missing)])
        self.assertEqual(exit_code, 2)
        self.assertIn("data file not found", logs.output[0])

    def test_missing_data_module_gives_clear_exit_code(self):
        with mock.patch.dict(sys.modules, {"data": None}):
            with self.assertLogs("experiments.c_inference", level="ERROR") as logs:
                exit_code = c_inference.main(["--results-dir", str(self.results_dir),
                                              "--data-path", str(self.data_file)])
        self.assertEqual(exit_code, 2)
        self.assertIn("data.py", logs.output[0])


if __name__ == "__main__":
    unittest.main()
