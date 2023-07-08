import os
import shutil
import unittest
from os.path import join
from pathlib import Path

from rsmtool.test_utils import copy_data_files

# allow test directory to be set via an environment variable
# which is needed for package testing
TEST_DIR = os.environ.get("TESTDIR", None)
if TEST_DIR:
    rsmtool_test_dir = TEST_DIR
else:
    from rsmtool.test_utils import rsmtool_test_dir


class TestCopyData(unittest.TestCase):
    dirs_to_remove = []

    @classmethod
    def tearDownClass(cls):
        for temp_dir in cls.dirs_to_remove:
            shutil.rmtree(temp_dir)

    def test_copy_data_files(self):
        file_dict = {
            "train": "data/files/train.csv",
            "features": "data/experiments/lr/features.csv",
        }
        expected_dict = {
            "train": join("temp_test_copy_data_file", "train.csv"),
            "features": join("temp_test_copy_data_file", "features.csv"),
        }
        self.dirs_to_remove.append("temp_test_copy_data_file")
        output_dict = copy_data_files("temp_test_copy_data_file", file_dict, rsmtool_test_dir)
        for file_type in expected_dict:
            self.assertEqual(output_dict[file_type], expected_dict[file_type])
            self.assertTrue(Path(output_dict[file_type]).exists())
            self.assertTrue(Path(output_dict[file_type]).is_file())

    def test_copy_data_files_directory(self):
        file_dict = {"exp_dir": "data/experiments/lr-self-compare/lr-subgroups"}
        expected_dict = {"exp_dir": join("temp_test_copy_dirs", "lr-subgroups")}
        self.dirs_to_remove.append("temp_test_copy_dirs")
        output_dict = copy_data_files("temp_test_copy_dirs", file_dict, rsmtool_test_dir)
        for file_type in expected_dict:
            self.assertEqual(output_dict[file_type], expected_dict[file_type])
            self.assertTrue(Path(output_dict[file_type]).exists())
            self.assertTrue(Path(output_dict[file_type]).is_dir())

    def test_copy_data_files_files_and_directories(self):
        file_dict = {
            "exp_dir": "data/experiments/lr-self-compare/lr-subgroups",
            "test": "data/files/test.csv",
        }
        expected_dict = {
            "exp_dir": join("temp_test_copy_mixed", "lr-subgroups"),
            "test": join("temp_test_copy_mixed", "test.csv"),
        }
        self.dirs_to_remove.append("temp_test_copy_mixed")
        output_dict = copy_data_files("temp_test_copy_mixed", file_dict, rsmtool_test_dir)
        for file_type in expected_dict:
            self.assertEqual(output_dict[file_type], expected_dict[file_type])
            self.assertTrue(Path(output_dict[file_type]).exists())
        self.assertTrue(Path(output_dict["exp_dir"]).is_dir())
        self.assertTrue(Path(output_dict["test"]).is_file())
