import unittest

import numpy as np

from config import DATA_PATH, N_TRAIN, N_TEST
from data import load_data, split, transform, recover_coef


class TestData(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.X, cls.y, cls.feature_names = load_data(DATA_PATH)

        (
            cls.X_train,
            cls.X_test,
            cls.y_train,
            cls.y_test,
        ) = split(cls.X, cls.y)

    def test_load_data_shapes(self):
        self.assertEqual(self.X.shape, (1030, 8))
        self.assertEqual(self.y.shape, (1030,))
        self.assertEqual(len(self.feature_names), 8)

    def test_split_shapes(self):
        self.assertEqual(self.X_train.shape, (N_TRAIN, 8))
        self.assertEqual(self.X_test.shape, (N_TEST, 8))
        self.assertEqual(self.y_train.shape, (N_TRAIN,))
        self.assertEqual(self.y_test.shape, (N_TEST,))

    def test_correct_test_rows(self):
        np.testing.assert_array_equal(
            self.X_test,
            self.X[500:630]
        )

        np.testing.assert_array_equal(
            self.y_test,
            self.y[500:630]
        )

    def test_standardize_training_data(self):
        X_std, params = transform(
            self.X_train,
            "standardize"
        )

        np.testing.assert_allclose(
            X_std.mean(axis=0),
            np.zeros(8),
            atol=1e-12
        )

        np.testing.assert_allclose(
            X_std.std(axis=0),
            np.ones(8),
            atol=1e-12
        )

        self.assertEqual(
            params["mode"],
            "standardize"
        )

    def test_test_set_uses_training_parameters(self):
        X_train_std, params = transform(
            self.X_train,
            "standardize"
        )

        X_test_std, _ = transform(
            self.X_test,
            "standardize",
            params=params
        )

        expected = (
            self.X_test - params["mean"]
        ) / params["std"]

        np.testing.assert_allclose(
            X_test_std,
            expected
        )

    def test_log1p(self):
        X_log, params = transform(
            self.X_train,
            "log1p"
        )

        np.testing.assert_allclose(
            X_log,
            np.log1p(self.X_train)
        )

        self.assertEqual(
            params["mode"],
            "log1p"
        )

    def test_recover_standardized_coefficients(self):
        X_std, params = transform(
            self.X_train,
            "standardize"
        )

        # Arbitrary model on standardized features
        theta_std = np.array(
            [10.0, 1.0, 2.0, 3.0, 4.0,
             5.0, 6.0, 7.0, 8.0]
        )

        theta_raw = recover_coef(
            theta_std,
            params
        )

        pred_std = (
            theta_std[0]
            + X_std @ theta_std[1:]
        )

        pred_raw = (
            theta_raw[0]
            + self.X_train @ theta_raw[1:]
        )

        np.testing.assert_allclose(
            pred_std,
            pred_raw,
            atol=1e-10
        )


if __name__ == "__main__":
    unittest.main()