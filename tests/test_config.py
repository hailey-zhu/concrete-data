"""Tests for config.py."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import config


class DataFileTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()

    def tearDown(self):
        self._tmp.cleanup()

    def test_configured_data_file_exists(self):
        """Smoke test: every experiment reads config.DATA_PATH."""
        try:
            config.require_data_file()
        except FileNotFoundError as exc:
            self.fail(str(exc))

    def test_existing_file_is_returned(self):
        data_file = self.root / "Concrete_Data.csv"
        data_file.write_text("", encoding="utf-8")
        self.assertEqual(config.require_data_file(data_file), data_file)

    def test_missing_file_message_names_the_files_that_exist(self):
        (self.root / "data").mkdir()
        (self.root / "data" / "Concrete_Data.csv").write_text("", encoding="utf-8")
        with mock.patch.object(config, "REPO_ROOT", self.root):
            with self.assertRaises(FileNotFoundError) as caught:
                config.require_data_file(self.root / "old_folder" / "Concrete_Data.xls")
        message = str(caught.exception)
        self.assertIn("old_folder/Concrete_Data.xls", message)
        self.assertIn("the repository contains data/Concrete_Data.csv", message)

    def test_missing_file_message_when_no_data_anywhere(self):
        with mock.patch.object(config, "REPO_ROOT", self.root):
            with self.assertRaisesRegex(FileNotFoundError, "no Concrete_Data"):
                config.require_data_file(self.root / "Concrete_Data.xls")


if __name__ == "__main__":
    unittest.main()
