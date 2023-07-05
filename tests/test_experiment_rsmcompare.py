import os
import tempfile
import unittest
from os import getcwd
from os.path import join

from nose2.tools import params

from rsmtool import run_comparison
from rsmtool.configuration_parser import Configuration
from rsmtool.test_utils import check_run_comparison, copy_data_files, do_run_comparison

# allow test directory to be set via an environment variable
# which is needed for package testing
TEST_DIR = os.environ.get("TESTDIR", None)
if TEST_DIR:
    rsmtool_test_dir = TEST_DIR
else:
    from rsmtool.test_utils import rsmtool_test_dir


class TestExperimentRsmcompare(unittest.TestCase):
    @params(
        {"source": "lr-self-compare", "experiment_id": "lr_subgroups_vs_lr_subgroups"},
        {"source": "lr-different-compare", "experiment_id": "lr_vs_lr_subset_features"},
        {"source": "lr-self-compare-with-h2", "experiment_id": "lr_with_h2_vs_lr_with_h2"},
        {
            "source": "lr-self-compare-with-custom-order",
            "experiment_id": "lr_subgroups_vs_lr_subgroups",
        },
        {
            "source": "lr-self-compare-with-chosen-sections",
            "experiment_id": "lr_subgroups_vs_lr_subgroups",
        },
        {
            "source": "lr-self-compare-with-custom-sections-and-custom-order",
            "experiment_id": "lr_subgroups_vs_lr_subgroups",
        },
        {
            "source": "lr-self-compare-with-thumbnails",
            "experiment_id": "lr_subgroups_vs_lr_subgroups",
        },
        {"source": "linearsvr-self-compare", "experiment_id": "LinearSVR_vs_LinearSVR"},
        {"source": "lr-eval-self-compare", "experiment_id": "lr_eval_with_h2_vs_lr_eval_with_h2"},
        {"source": "lr-eval-tool-compare", "experiment_id": "lr_with_h2_vs_lr_eval_with_h2"},
        {
            "source": "lr-self-compare-different-format",
            "experiment_id": "lr_with_tsv_output_vs_lr_with_tsv_output",
        },
        {
            "source": "lr-self-compare-with-subgroups-and-h2",
            "experiment_id": "lr-subgroups-with-h2_vs_lr-subgroups-with-h2",
        },
        {
            "source": "lr-self-compare-with-subgroups-and-edge-cases",
            "experiment_id": "lr-subgroups-with-edge-cases_vs_lr-subgroups-with-edge-cases",
        },
    )
    def test_run_experiment_parameterized(self, kwargs):
        if TEST_DIR:
            kwargs["given_test_dir"] = TEST_DIR
        check_run_comparison(**kwargs)

    def test_run_experiment_lr_compare_with_object(self):
        """Test rsmcompare using the Configuration object, rather than a file."""
        source = "lr-self-compare-object"
        experiment_id = "lr_self_compare_object"

        configdir = join(rsmtool_test_dir, "data", "experiments", source)

        config_dict = {
            "comparison_id": "lr_self_compare_object",
            "experiment_dir_old": "lr-subgroups",
            "experiment_id_old": "lr_subgroups",
            "description_old": "Using all features with a LinearRegression model.",
            "use_scaled_predictions_old": True,
            "experiment_dir_new": "lr-subgroups",
            "experiment_id_new": "lr_subgroups",
            "description_new": "Using all features with a LinearRegression model.",
            "use_scaled_predictions_new": True,
            "subgroups": ["QUESTION"],
        }

        config_obj = Configuration(config_dict, context="rsmcompare", configdir=configdir)

        check_run_comparison(source, experiment_id, config_obj_or_dict=config_obj)

    def test_run_experiment_lr_compare_with_dictionary(self):
        """Test rsmcompare using the dictionary object, rather than a file."""
        source = "lr-self-compare-dict"
        experiment_id = "lr_self_compare_dict"

        # set up a temporary directory since
        # we will be using getcwd
        temp_dir = tempfile.TemporaryDirectory(prefix=getcwd())

        old_file_dict = {"experiment_dir": "data/experiments/lr-self-compare-dict/lr-subgroups"}

        new_file_dict = copy_data_files(temp_dir.name, old_file_dict, rsmtool_test_dir)

        config_dict = {
            "comparison_id": "lr_self_compare_dict",
            "experiment_dir_old": new_file_dict["experiment_dir"],
            "experiment_id_old": "lr_subgroups",
            "description_old": "Using all features with a LinearRegression model.",
            "use_scaled_predictions_old": True,
            "experiment_dir_new": new_file_dict["experiment_dir"],
            "experiment_id_new": "lr_subgroups",
            "description_new": "Using all features with a LinearRegression model.",
            "use_scaled_predictions_new": True,
            "subgroups": ["QUESTION"],
        }

        check_run_comparison(source, experiment_id, config_obj_or_dict=config_dict)

    def test_run_comparison_wrong_input_format(self):
        config_list = [("experiment_id", "AAAA"), ("train_file", "some_path")]
        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory() as temp_dir:
                run_comparison(config_list, temp_dir)

    def test_run_experiment_lr_compare_wrong_directory(self):
        # basic rsmcompare experiment comparing a LinearRegression
        # experiment to itself
        source = "lr-self-compare-wrong-directory"
        config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmcompare.json")
        with self.assertRaises(FileNotFoundError):
            do_run_comparison(source, config_file)

    def test_run_experiment_lr_compare_wrong_experiment_id(self):
        # basic rsmcompare experiment comparing a LinearRegression
        # experiment to itself
        source = "lr-self-compare-wrong-id"
        config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmcompare.json")
        with self.assertRaises(FileNotFoundError):
            do_run_comparison(source, config_file)
