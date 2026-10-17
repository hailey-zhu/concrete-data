"""Tests for result_schema.py."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from result_schema import (
    ResultSchemaError, config_label, fields_for_mode, make_result, read_result,
    validate_result, write_result,
)


def valid_fields(**overrides):
    fields = dict(
        exp_id="a_gd_standardize", owner="A", transform="none", scaling="standardize",
        alpha=0.1, n_iters=5000, converged_iter=1445,
        theta=[35.0, 1.0, 2.0], train_mse=104.1, test_mse=141.2, train_r2=0.62, test_r2=0.37,
        p_value=None, solver="gd", features=["Cement", "Age"],
    )
    fields.update(overrides)
    return fields


class MakeResultTests(unittest.TestCase):
    def test_converts_numpy_types_to_builtins(self):
        record = make_result(**valid_fields(theta=np.array([1.0, 2.0, 3.0]),
                                            n_iters=np.int64(5000), train_mse=np.float64(1.5)))
        self.assertEqual(record["theta"], [1.0, 2.0, 3.0])
        self.assertIs(type(record["n_iters"]), int)
        json.dumps(record)  # must be serialisable

    def test_closed_form_result_without_gd_fields(self):
        record = make_result(**valid_fields(exp_id="c_ols_raw", owner="C", alpha=None, n_iters=None,
                                            converged_iter=None, solver="statsmodels",
                                            p_value=[0.1, 0.01, 0.5]))
        self.assertIsNone(record["alpha"])

    def test_nan_metrics_allowed_for_diverged_runs(self):
        make_result(**valid_fields(train_mse=float("inf"), train_r2=float("-inf"),
                                   test_mse=float("nan"), test_r2=float("nan")))

    def test_accepts_output_of_fields_for_mode(self):
        fields = valid_fields()
        del fields["transform"], fields["scaling"]
        record = make_result(**fields, **fields_for_mode("log1p"))
        self.assertEqual((record["transform"], record["scaling"]), ("log1p", "none"))


class ValidationTests(unittest.TestCase):
    def assertInvalid(self, record, pattern):
        with self.assertRaisesRegex(ResultSchemaError, pattern):
            validate_result(record)

    def test_valid_record_passes(self):
        validate_result(valid_fields())

    def test_missing_required_field(self):
        record = valid_fields()
        del record["test_r2"]
        self.assertInvalid(record, "missing required field.*test_r2")

    def test_unknown_field(self):
        self.assertInvalid(valid_fields(trian_mse=1.0), "unknown field.*trian_mse")

    def test_bad_choices(self):
        self.assertInvalid(valid_fields(owner="D"), "owner")
        self.assertInvalid(valid_fields(transform="log"), "transform")
        self.assertInvalid(valid_fields(solver="adam"), "solver")

    def test_exp_id_must_carry_owner_prefix(self):
        self.assertInvalid(valid_fields(exp_id="b_gd_standardize"), "start with 'a_'")

    def test_exp_id_characters(self):
        self.assertInvalid(valid_fields(exp_id="a_GD run"), "exp_id")

    def test_p_value_length_must_match_theta(self):
        self.assertInvalid(valid_fields(p_value=[0.1, 0.2]), "p_value has 2 entries")

    def test_p_value_must_be_probabilities(self):
        self.assertInvalid(valid_fields(p_value=[0.1, 1.2, 0.3]), "probabilities")

    def test_features_length_must_match_weights(self):
        self.assertInvalid(valid_fields(features=["Cement"]), "features has 1 names")

    def test_converged_iter_cannot_exceed_n_iters(self):
        self.assertInvalid(valid_fields(converged_iter=6000), "exceeds n_iters")

    def test_r2_cannot_exceed_one(self):
        self.assertInvalid(valid_fields(test_r2=1.5), "test_r2")

    def test_bool_is_not_a_number(self):
        self.assertInvalid(valid_fields(n_iters=True), "n_iters")

    def test_reports_every_problem_at_once(self):
        with self.assertRaises(ResultSchemaError) as caught:
            validate_result(valid_fields(owner="D", alpha=-1.0, test_r2=2.0))
        message = str(caught.exception)
        for name in ("owner", "alpha", "test_r2"):
            self.assertIn(name, message)


class FileTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_round_trip(self):
        record = make_result(**valid_fields())
        path = write_result(record, self.dir)
        self.assertEqual(path, self.dir / "a_gd_standardize.json")
        self.assertEqual(read_result(path), record)

    def test_rewrite_replaces_previous_result(self):
        write_result(make_result(**valid_fields(train_mse=1.0)), self.dir)
        path = write_result(make_result(**valid_fields(train_mse=2.0)), self.dir)
        self.assertEqual(read_result(path)["train_mse"], 2.0)
        self.assertEqual(len(list(self.dir.iterdir())), 1)  # no temporary files left behind

    def test_optional_fields_default_to_none_when_absent(self):
        record = valid_fields()
        del record["solver"], record["features"]
        path = self.dir / "a_gd_standardize.json"
        path.write_text(json.dumps(record), encoding="utf-8")
        loaded = read_result(path)
        self.assertIsNone(loaded["solver"])
        self.assertIsNone(loaded["features"])

    def test_file_name_must_match_exp_id(self):
        path = self.dir / "a_other_name.json"
        path.write_text(json.dumps(valid_fields()), encoding="utf-8")
        with self.assertRaisesRegex(ResultSchemaError, "does not match the file name"):
            read_result(path)

    def test_invalid_json(self):
        path = self.dir / "a_broken.json"
        path.write_text("{not json", encoding="utf-8")
        with self.assertRaisesRegex(ResultSchemaError, "a_broken.json: not valid JSON"):
            read_result(path)


class LabelTests(unittest.TestCase):
    def test_modes_map_to_labels(self):
        labels = [config_label(**fields_for_mode(mode)) for mode in ("raw", "standardize", "log1p")]
        self.assertEqual(labels, ["Untransformed", "Standardized", "Log-transformed"])

    def test_unknown_mode(self):
        with self.assertRaisesRegex(ValueError, "unknown transform mode"):
            fields_for_mode("sqrt")


if __name__ == "__main__":
    unittest.main()
