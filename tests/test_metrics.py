"""Tests for metrics.py."""
import unittest

import numpy as np

from metrics import add_intercept, check_inputs, evaluate, evaluate_split, mse, predict, r_squared
from result_schema import METRIC_FIELDS


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.X = rng.normal(size=(50, 3))
        self.theta = np.array([1.0, 2.0, -1.0, 0.5])
        self.y = predict(self.X, self.theta)

    def test_perfect_fit(self):
        result = evaluate(self.X, self.y, self.theta)
        self.assertAlmostEqual(result["mse"], 0.0)
        self.assertAlmostEqual(result["r2"], 1.0)

    def test_hand_computed_values(self):
        # predictions 1, 2, 3 vs targets 1, 2, 4: MSE = 1/3, Var(y) = 14/9, R² = 11/14
        result = evaluate([0.0, 1.0, 2.0], [1.0, 2.0, 4.0], [1.0, 1.0])
        self.assertAlmostEqual(result["mse"], 1 / 3)
        self.assertAlmostEqual(result["r2"], 11 / 14)

    def test_predicting_the_mean_gives_zero_r2(self):
        noisy = self.y + np.random.default_rng(1).normal(size=self.y.shape)
        theta = np.array([noisy.mean(), 0.0, 0.0, 0.0])
        result = evaluate(self.X, noisy, theta)
        self.assertAlmostEqual(result["r2"], 0.0)
        self.assertAlmostEqual(result["mse"], np.var(noisy))

    def test_accepts_1d_feature_for_univariate_models(self):
        x = self.X[:, 0]
        theta = [0.5, 2.0]
        self.assertEqual(evaluate(x, 0.5 + 2.0 * x, theta)["r2"], 1.0)

    def test_returns_python_floats(self):
        result = evaluate(self.X, self.y, self.theta)
        self.assertIs(type(result["mse"]), float)
        self.assertIs(type(result["r2"]), float)

    def test_diverged_theta_gives_non_finite_metrics_without_raising(self):
        theta = np.array([np.inf, 1e308, 1e308, 1e308])
        result = evaluate(self.X, self.y, theta)
        self.assertFalse(np.isfinite(result["mse"]))
        self.assertFalse(np.isfinite(result["r2"]))

    def test_evaluate_split_keys_match_result_schema(self):
        result = evaluate_split(self.X, self.y, self.X[:10], self.y[:10], self.theta)
        self.assertEqual(tuple(result), METRIC_FIELDS)


class ValidationTests(unittest.TestCase):
    def test_theta_length_must_be_n_features_plus_one(self):
        with self.assertRaisesRegex(ValueError, "n_features \\+ 1"):
            evaluate(np.ones((5, 2)), np.arange(5.0), [0.0, 1.0])

    def test_sample_counts_must_match(self):
        with self.assertRaisesRegex(ValueError, "samples"):
            evaluate(np.ones((5, 2)), np.arange(4.0), [0.0, 1.0, 1.0])

    def test_non_finite_features_rejected(self):
        X = np.ones((5, 1))
        X[2, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN"):
            evaluate(X, np.arange(5.0), [0.0, 1.0])

    def test_constant_target_has_undefined_r2(self):
        with self.assertRaisesRegex(ValueError, "zero variance"):
            r_squared([3.0, 3.0, 3.0], [1.0, 2.0, 3.0])

    def test_mse_shape_mismatch(self):
        with self.assertRaises(ValueError):
            mse([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_check_inputs_flattens_column_vector_target(self):
        X, y, theta = check_inputs(np.ones((4, 2)), np.arange(4.0).reshape(-1, 1))
        self.assertEqual(X.shape, (4, 2))
        self.assertEqual(y.shape, (4,))
        self.assertIsNone(theta)

    def test_add_intercept_prepends_ones(self):
        design = add_intercept([[2.0], [3.0]])
        np.testing.assert_array_equal(design, [[1.0, 2.0], [1.0, 3.0]])


if __name__ == "__main__":
    unittest.main()
