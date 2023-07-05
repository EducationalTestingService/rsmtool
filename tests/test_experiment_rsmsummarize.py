import os
import tempfile
import unittest
from os import getcwd
from os.path import join

from nose2.tools import params

from rsmtool import run_summary
from rsmtool.configuration_parser import Configuration
from rsmtool.test_utils import check_run_summary, copy_data_files, do_run_summary

# allow test directory to be set via an environment variable
# which is needed for package testing
TEST_DIR = os.environ.get("TESTDIR", None)
if TEST_DIR:
    rsmtool_test_dir = TEST_DIR
else:
    from rsmtool.test_utils import rsmtool_test_dir


class TestExperimentRsmsummarize(unittest.TestCase):
    @params(
        {"source": "lr-self-summary"},
        {"source": "linearsvr-self-summary"},
        {"source": "lr-self-eval-summary"},
        {"source": "lr-self-summary-with-custom-sections"},
        {"source": "lr-self-summary-with-tsv-inputs"},
        {"source": "lr-self-summary-with-tsv-output", "file_format": "tsv"},
        {"source": "lr-self-summary-with-xlsx-output", "file_format": "xlsx"},
        {"source": "lr-self-summary-no-scaling"},
        {"source": "lr-self-summary-with-h2"},
        {"source": "summary-with-custom-names"},
        {"source": "lr-self-summary-no-trim"},
    )
    def test_run_experiment_parameterized(self, kwargs):
        if TEST_DIR:
            kwargs["given_test_dir"] = TEST_DIR
        check_run_summary(**kwargs)

    def test_run_experiment_lr_summary_with_object(self):
        """
        test rsmsummarize using the Configuration object, rather than a file;
        """

        source = "lr-self-summary-object"

        # we set the configuration directory to point to the test directory
        # to ensure that the results are identical to what we would expect
        # if we had run this test with a configuration file instead
        configdir = join(rsmtool_test_dir, "data", "experiments", source)

        config_dict = {
            "summary_id": "model_comparison",
            "experiment_dirs": ["lr-subgroups", "lr-subgroups", "lr-subgroups"],
            "description": "Comparison of rsmtool experiment with itself.",
        }

        config_obj = Configuration(config_dict, context="rsmsummarize", configdir=configdir)

        check_run_summary(source, config_obj_or_dict=config_obj)

    def test_run_experiment_lr_summary_dictionary(self):
        """
        Test rsmsummarize using the dictionary object, rather than a file;
        """
        source = "lr-self-summary-dict"

        # set up a temporary directory since
        # we will be using getcwd
        temp_dir = tempfile.TemporaryDirectory(prefix=getcwd())

        old_file_dict = {"experiment_dir": "data/experiments/lr-self-summary-dict/lr-subgroups"}

        new_file_dict = copy_data_files(temp_dir.name, old_file_dict, rsmtool_test_dir)

        config_dict = {
            "summary_id": "model_comparison",
            "experiment_dirs": [
                new_file_dict["experiment_dir"],
                new_file_dict["experiment_dir"],
                new_file_dict["experiment_dir"],
            ],
            "description": "Comparison of rsmtool experiment with itself.",
        }

        check_run_summary(source, config_obj_or_dict=config_dict)

    def test_run_summary_wrong_input_format(self):
        config_list = [("experiment_id", "AAAA"), ("train_file", "some_path")]
        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory() as temp_dir:
                run_summary(config_list, temp_dir)

    def test_run_experiment_summary_wrong_directory(self):
        # rsmsummarize experiment where the specified directory
        # does not exist
        source = "summary-wrong-directory"
        config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmsummarize.json")
        with self.assertRaises(FileNotFoundError):
            do_run_summary(source, config_file)

    def test_run_experiment_summary_no_csv_directory(self):
        # rsmsummarize experiment where the specified directory
        # does not contain any rsmtool experiments
        source = "summary-no-output-dir"
        config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmsummarize.json")
        with self.assertRaises(FileNotFoundError):
            do_run_summary(source, config_file)

    def test_run_experiment_summary_no_json(self):
        # rsmsummarize experiment where the specified directory
        # does not contain any json files
        source = "summary-no-json-file"
        config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmsummarize.json")
        with self.assertRaises(FileNotFoundError):
            do_run_summary(source, config_file)

    def test_run_experiment_summary_too_many_jsons(self):
        # rsmsummarize experiment where the specified directory
        # does contains several jsons files and the user
        # specified experiment names
        source = "summary-too-many-jsons"
        config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmsummarize.json")
        with self.assertRaises(ValueError):
            do_run_summary(source, config_file)
