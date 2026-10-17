"""Tests for inference.py."""
import unittest
import warnings

import numpy as np

from inference import NonOptimalThetaWarning, fit_ols_statsmodels, ols_inference
from metrics import add_intercept


def _synthetic_data(n=200, seed=0):
    """y depends strongly on x1 and x3, not at all on x2; x1 and x2 are correlated."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = 0.6 * x1 + rng.normal(size=n)
    x3 = rng.uniform(0, 10, size=n)
    X = np.column_stack([x1, x2, x3])
    y = 2.0 + 1.5 * x1 + 0.0 * x2 - 0.7 * x3 + rng.normal(scale=1.0, size=n)
    return X, y


class OlsInferenceTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = _synthetic_data()
        self.ols = fit_ols_statsmodels(self.X, self.y)

    def test_matches_statsmodels(self):
        manual = ols_inference(self.X, self.y, self.ols["theta"])
        for key in ("se", "t_stat", "p_value"):
            np.testing.assert_allclose(manual[key], self.ols[key], rtol=1e-8, err_msg=key)

    def test_matches_textbook_formula(self):
        design = add_intercept(self.X)
        residuals = self.y - design @ self.ols["theta"]
        sigma2 = residuals @ residuals / (len(self.y) - design.shape[1])
        expected_se = np.sqrt(np.diag(sigma2 * np.linalg.inv(design.T @ design)))
        np.testing.assert_allclose(ols_inference(self.X, self.y, self.ols["theta"])["se"],
                                   expected_se, rtol=1e-8)

    def test_separates_real_and_null_effects(self):
        p = self.ols["p_value"]
        self.assertLess(p[1], 1e-10)     # x1: real effect
        self.assertGreater(p[2], 0.01)   # x2: no effect
        self.assertLess(p[3], 1e-10)     # x3: real effect

    def test_output_is_aligned_with_theta(self):
        result = ols_inference(self.X, self.y, self.ols["theta"])
        for key in ("se", "t_stat", "p_value"):
            self.assertEqual(result[key].shape, (4,))

    def test_linear_rescaling_keeps_slope_pvalues(self):
        scaled = (self.X - self.X.mean(axis=0)) / self.X.std(axis=0)
        p_raw = self.ols["p_value"]
        p_scaled = fit_ols_statsmodels(scaled, self.y)["p_value"]
        np.testing.assert_allclose(p_scaled[1:], p_raw[1:], rtol=1e-6)
        # the intercept's meaning changes, so its p-value moves by orders of magnitude
        self.assertGreater(abs(np.log10(p_scaled[0]) - np.log10(p_raw[0])), 1.0)

    def test_univariate_1d_input(self):
        result = fit_ols_statsmodels(self.X[:, 0], self.y)
        self.assertEqual(result["theta"].shape, (2,))


class OptimalityWarningTests(unittest.TestCase):
    def setUp(self):
        self.X, self.y = _synthetic_data()
        self.theta = fit_ols_statsmodels(self.X, self.y)["theta"]

    def test_non_optimal_theta_warns(self):
        with self.assertWarns(NonOptimalThetaWarning):
            ols_inference(self.X, self.y, self.theta + 0.5)

    def test_optimal_theta_does_not_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", NonOptimalThetaWarning)
            ols_inference(self.X, self.y, self.theta)


class InvalidInputTests(unittest.TestCase):
    def test_collinear_features_rejected(self):
        X, y = _synthetic_data()
        X = np.column_stack([X, 2 * X[:, 0]])
        with self.assertRaisesRegex(ValueError, "rank-deficient"):
            fit_ols_statsmodels(X, y)

    def test_too_few_samples_rejected(self):
        with self.assertRaisesRegex(ValueError, "more samples than parameters"):
            ols_inference(np.ones((2, 1)), [1.0, 2.0], [0.0, 1.0])

    def test_non_finite_theta_rejected(self):
        X, y = _synthetic_data()
        with self.assertRaisesRegex(ValueError, "diverge"):
            ols_inference(X, y, [np.nan, 0.0, 0.0, 0.0])

    def test_exact_fit_rejected(self):
        x = np.arange(10.0)
        with self.assertRaisesRegex(ValueError, "zero"):
            ols_inference(x, 3.0 + 2.0 * x, [3.0, 2.0])


if __name__ == "__main__":
    unittest.main()
