"""Tests for aggregate.py, driven by hand-made result files (division plan: "fake data first")."""
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

import aggregate
from result_schema import ResultSchemaError

FEATURES = ["Cement", "Water", "Age"]


def _record(exp_id, owner, transform="none", scaling="none", **overrides):
    record = {
        "exp_id": exp_id, "owner": owner, "transform": transform, "scaling": scaling,
        "alpha": None, "n_iters": None, "theta": [1.0, 0.1, -0.2, 0.3],
        "train_mse": 100.0, "test_mse": 140.0, "train_r2": 0.62, "test_r2": 0.37,
        "converged_iter": None, "p_value": None, "solver": None, "features": FEATURES,
    }
    record.update(overrides)
    return record


HAND_MADE_RESULTS = [
    _record("c_ols_raw", "C", solver="statsmodels", p_value=[0.07, 1e-40, 0.002, 1e-73]),
    _record("c_ols_standardize", "C", scaling="standardize", solver="statsmodels",
            p_value=[0.0, 1e-40, 0.002, 1e-73]),
    _record("c_ols_log1p", "C", transform="log1p", solver="statsmodels",
            train_mse=55.8, test_mse=54.4, train_r2=0.80, test_r2=0.76,
            p_value=[0.29, 1e-85, 1e-20, 1e-192]),
    # Person A: a multivariate GD run (no p-values)
    _record("a_gd_standardize", "A", scaling="standardize", solver="gd",
            alpha=0.1, n_iters=5000, converged_iter=1445),
    # Person B: univariate GD runs; two for Cement (the better one must be picked)
    _record("b_uni_cement_a1", "B", solver="gd", alpha=1e-6, n_iters=5000,
            theta=[0.0, 0.1], features=["Cement"], train_r2=0.10),
    _record("b_uni_cement_a2", "B", solver="gd", alpha=1e-5, n_iters=5000,
            theta=[0.0, 0.1], features=["Cement"], train_r2=0.17),
    # no solver recorded, but a learning rate: still counted as gradient descent
    _record("b_uni_age", "B", alpha=1e-4, n_iters=5000, converged_iter=900,
            theta=[0.0, 0.1], features=["Age"], train_r2=-0.19),
    # Person B: optimal (scikit-learn) univariate fits and the raw multivariate model
    _record("b_sklearn_univariate_f1", "B", solver="sklearn", theta=[0.0, 0.12],
            features=["Cement"]),
    _record("b_sklearn_univariate_f2", "B", solver="sklearn", theta=[0.0, -0.30],
            features=["Water"]),
    _record("b_sklearn_univariate_f3", "B", solver="sklearn", theta=[0.0, 0.09],
            features=["Age"]),
    _record("b_sklearn_multivariate_raw", "B", solver="sklearn",
            theta=[10.0, 0.10, -0.25, 0.08]),
]


class AggregateTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.results_dir, self.tables_dir, self.figures_dir = (
            root / "results", root / "tables", root / "figures")
        self.results_dir.mkdir()
        for record in HAND_MADE_RESULTS:
            self.write_json(record)

    def tearDown(self):
        self._tmp.cleanup()

    def write_json(self, record, name=None):
        path = self.results_dir / f"{name or record['exp_id']}.json"
        path.write_text(json.dumps(record), encoding="utf-8")


class CollectTests(AggregateTestCase):
    def test_one_row_per_file_indexed_by_exp_id(self):
        results = aggregate.collect(self.results_dir)
        self.assertEqual(len(results), len(HAND_MADE_RESULTS))
        self.assertEqual(list(results.index), sorted(r["exp_id"] for r in HAND_MADE_RESULTS))
        self.assertEqual(results.loc["c_ols_log1p", "config"], "Log-transformed")
        self.assertEqual(results.loc["a_gd_standardize", "converged_iter"], 1445)

    def test_empty_directory(self):
        empty = self.results_dir.parent / "empty"
        empty.mkdir()
        self.assertTrue(aggregate.collect(empty).empty)

    def test_missing_directory(self):
        with self.assertRaises(FileNotFoundError):
            aggregate.collect(self.results_dir / "nope")

    def test_invalid_file_strict_and_lenient(self):
        self.write_json(_record("c_bad", "Z"))
        with self.assertRaises(ResultSchemaError):
            aggregate.collect(self.results_dir)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            results = aggregate.collect(self.results_dir, strict=False)
        self.assertNotIn("c_bad", results.index)
        self.assertTrue(any("c_bad" in str(w.message) for w in caught))


class TableTests(AggregateTestCase):
    def setUp(self):
        super().setUp()
        self.results = aggregate.collect(self.results_dir)

    def test_summary_table(self):
        table = aggregate.summary_table(self.results)
        self.assertEqual(list(table.index), ["Untransformed", "Standardized", "Log-transformed"])
        self.assertEqual(list(table.columns),
                         ["Training MSE", "Testing MSE", "Training R²", "Testing R²"])
        self.assertAlmostEqual(table.loc["Log-transformed", "Testing R²"], 0.76)

    def test_summary_table_reports_missing_experiments(self):
        with self.assertRaisesRegex(aggregate.MissingResultError, "c_ols_sqrt"):
            aggregate.summary_table(self.results, ["c_ols_raw", "c_ols_sqrt"])

    def test_summary_table_rejects_duplicate_configurations(self):
        with self.assertRaisesRegex(ValueError, "share a configuration"):
            aggregate.summary_table(self.results, ["c_ols_standardize", "a_gd_standardize"])

    def test_pvalue_table(self):
        table = aggregate.pvalue_table(self.results)
        self.assertEqual(list(table.index), ["Intercept", *FEATURES])
        self.assertEqual(list(table.columns), ["Untransformed", "Standardized", "Log-transformed"])
        self.assertEqual(table.loc["Age", "Log-transformed"], 1e-192)

    def test_pvalue_table_requires_pvalues(self):
        with self.assertRaisesRegex(ValueError, "no p-values"):
            aggregate.pvalue_table(self.results, ["a_gd_standardize"])

    def test_univariate_table_keeps_best_run_per_feature(self):
        table = aggregate.univariate_table(self.results)
        self.assertEqual(list(table.index), ["Cement", "Age"])  # easiest first
        self.assertEqual(table.loc["Cement", "Experiment"], "b_uni_cement_a2")

    def test_univariate_table_empty_without_univariate_runs(self):
        only_ols = self.results.loc[list(aggregate.DEFAULT_OLS_IDS)]
        self.assertTrue(aggregate.univariate_table(only_ols).empty)

    def test_coefficient_comparison_table(self):
        table = aggregate.coefficient_comparison_table(self.results)
        self.assertEqual(list(table.index), FEATURES)
        self.assertAlmostEqual(table.loc["Cement", "Univariate coefficient"], 0.12)
        self.assertAlmostEqual(table.loc["Cement", "Multivariate coefficient"], 0.10)
        self.assertAlmostEqual(table.loc["Water", "Difference"], -0.30 - (-0.25))

    def test_coefficient_comparison_uses_optimal_fits_not_gradient_descent(self):
        # b_uni_cement_* share the configuration but were fitted by gradient descent
        table = aggregate.coefficient_comparison_table(self.results)
        self.assertAlmostEqual(table.loc["Cement", "Univariate coefficient"], 0.12)

    def test_coefficient_comparison_marks_missing_univariate_models(self):
        (self.results_dir / "b_sklearn_univariate_f2.json").unlink()
        results = aggregate.collect(self.results_dir)
        with self.assertLogs("aggregate", level="WARNING"):
            table = aggregate.coefficient_comparison_table(results)
        self.assertTrue(np.isnan(table.loc["Water", "Univariate coefficient"]))

    def test_coefficient_comparison_requires_the_multivariate_model(self):
        with self.assertRaises(aggregate.MissingResultError):
            aggregate.coefficient_comparison_table(self.results, "b_sklearn_absent")

    def test_trend_table(self):
        table = aggregate.trend_table(aggregate.pvalue_table(self.results),
                                      aggregate.univariate_table(self.results))
        self.assertEqual(table.loc["Age", "p-value (Untransformed multivariate)"], 1e-73)


class OutputTests(AggregateTestCase):
    def test_markdown_formatting(self):
        table = aggregate.pvalue_table(aggregate.collect(self.results_dir))
        text = aggregate.to_markdown(table, aggregate.format_pvalue)
        lines = text.splitlines()
        self.assertEqual(lines[0], "| Term | Untransformed | Standardized | Log-transformed |")
        self.assertEqual(lines[1], "| :-- | --: | --: | --: |")
        self.assertIn("| Intercept | 0.07 | <1e-300 | 0.29 |", lines)

    def test_markdown_escapes_pipes_and_shows_missing_values(self):
        results = aggregate.collect(self.results_dir)
        text = aggregate.to_markdown(aggregate.overview_table(results).assign(Note="a|b"))
        self.assertIn(r"a\|b", text)
        self.assertIn("—", text)

    def test_build_report_writes_every_output(self):
        written = aggregate.build_report(self.results_dir, self.tables_dir, self.figures_dir)
        names = {path.name for path in written}
        expected_tables = ["all_experiments", "optimal_models_summary",
                           "pvalues_by_configuration", "univariate_gd_training",
                           "significance_vs_univariate_training",
                           "univariate_vs_multivariate_coefficients"]
        for stem in expected_tables:
            self.assertIn(f"{stem}.csv", names)
            self.assertIn(f"{stem}.md", names)
        self.assertIn("pvalues_by_configuration.png", names)
        self.assertTrue(all(path.is_file() for path in written))

    def test_build_report_without_ols_results_still_writes_overview(self):
        for exp_id in aggregate.DEFAULT_OLS_IDS:
            (self.results_dir / f"{exp_id}.json").unlink()
        with self.assertLogs("aggregate", level="WARNING"):
            written = aggregate.build_report(self.results_dir, self.tables_dir, self.figures_dir)
        names = {path.name for path in written}
        self.assertIn("all_experiments.csv", names)
        self.assertNotIn("pvalues_by_configuration.csv", names)

    def test_csv_keeps_full_precision(self):
        aggregate.build_report(self.results_dir, self.tables_dir, self.figures_dir)
        csv_path = self.tables_dir / "pvalues_by_configuration.csv"
        csv_text = csv_path.read_text(encoding="utf-8-sig")
        self.assertIn("1e-192", csv_text)

    def test_nice_upper_bound(self):
        self.assertEqual([aggregate._nice_upper_bound(v) for v in (0.5, 3, 191.8, 1000)],
                         [2, 5, 200, 1000])


if __name__ == "__main__":
    unittest.main()
