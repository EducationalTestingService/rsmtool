import os
import tempfile
import unittest
from os import getcwd
from os.path import join
from unittest.mock import patch

from nose2.tools import params

from rsmtool.configuration_parser import Configuration
from rsmtool.test_utils import check_run_explain, copy_data_files, do_run_explain

# allow test directory to be set via an environment variable
# which is needed for package testing
TEST_DIR = os.environ.get("TESTDIR", None)
if TEST_DIR:
    rsmtool_test_dir = TEST_DIR
else:
    from rsmtool.test_utils import rsmtool_test_dir  # noqa: F401


class TestExperimentRsmexplain(unittest.TestCase):
    @params(
        {"source": "svr-explain", "experiment_id": "svr"},
        {"source": "knn-explain", "experiment_id": "knn"},
        {"source": "rf-explain", "experiment_id": "rf"},
    )
    def test_run_experiment_parameterized(self, kwargs):
        if TEST_DIR:
            kwargs["given_test_dir"] = TEST_DIR
        check_run_explain(**kwargs)

    def test_run_experiment_svr_explain_with_object(self):
        """Test rsmexplain using a Configuration object, rather than a file."""
        source = "svr-explain-object"
        experiment_id = "svr_explain_object"

        configdir = join(rsmtool_test_dir, "data", "experiments", source)

        config_dict = {
            "description": "Explaning an SVR model trained on all features.",
            "experiment_dir": "existing_experiment",
            "experiment_id": "svr_explain_object",
            "background_data": "../../files/train.csv",
            "background_kmeans_size": 50,
            "explain_data": "../../files/test.csv",
            "id_column": "ID",
            "sample_size": 10,
            "num_features_to_display": 15,
            "show_auto_cohorts": True,
            "truncate_outliers": False,
        }

        config_obj = Configuration(config_dict, context="rsmexplain", configdir=configdir)

        check_run_explain(source, experiment_id, config_obj_or_dict=config_obj)

    def test_run_experiment_svr_explain_with_dictionary(self):
        """Test rsmexplain using the dictionary object, rather than a file."""
        source = "svr-explain-dict"
        experiment_id = "svr_explain_dict"

        # set up a temporary directory since
        # we will be using getcwd
        temp_dir = tempfile.TemporaryDirectory(prefix=getcwd())

        old_file_dict = {
            "experiment_dir": "data/experiments/svr-explain-dict/existing_experiment",
            "background_data": "data/files/train.csv",
            "explain_data": "data/files/test.csv",
        }

        new_file_dict = copy_data_files(temp_dir.name, old_file_dict, rsmtool_test_dir)

        config_dict = {
            "description": "Explaning an SVR model trained on all features.",
            "experiment_dir": new_file_dict["experiment_dir"],
            "experiment_id": "svr_explain_dict",
            "background_data": new_file_dict["background_data"],
            "background_kmeans_size": 50,
            "explain_data": new_file_dict["explain_data"],
            "id_column": "ID",
            "sample_ids": "RESPONSE_6, RESPONSE_11, RESPONSE_21",
            "num_features_to_display": 15,
            "show_auto_cohorts": True,
        }

        check_run_explain(source, experiment_id, config_obj_or_dict=config_dict)

    def test_run_rsmexplain_different_standardize_features_value(self):
        """Check that rsmtool standardize features value overrides rsmexplain value."""
        # set up a temporary directory since we will be using getcwd
        temp_dir = tempfile.TemporaryDirectory(prefix=getcwd())

        old_file_dict = {
            "experiment_dir": "data/experiments/knn-explain-diff-std/existing_experiment",
            "background_data": "data/files/train.csv",
            "explain_data": "data/files/test.csv",
        }

        new_file_dict = copy_data_files(temp_dir.name, old_file_dict, rsmtool_test_dir)

        source = "knn-explain-diff-std"
        config_dict = {
            "description": "Explaning an KNeighborsRegressor model trained on all features.",
            "experiment_dir": new_file_dict["experiment_dir"],
            "experiment_id": "knn_diff_std",
            "background_data": new_file_dict["background_data"],
            "background_kmeans_size": 50,
            "explain_data": new_file_dict["explain_data"],
            "standardize_features": False,
            "id_column": "ID",
            "sample_size": 10,
            "num_features_to_display": 15,
            "show_auto_cohorts": True,
        }

        # check `standardize_features` in the config has been overridden to `True`
        # since that was the value in rsmtool configuration
        with patch("rsmtool.rsmexplain.generate_report") as mock_generate_report:
            do_run_explain(source, config_dict)
            called_config = mock_generate_report.call_args[0][3]
            self.assertEqual(called_config["standardize_features"], True)

    def test_run_rsmexplain_same_standardize_features_value(self):
        """Check that rsmexplain standardize features value does not change if matching rsmtool."""
        # set up a temporary directory since we will be using getcwd
        temp_dir = tempfile.TemporaryDirectory(prefix=getcwd())

        old_file_dict = {
            "experiment_dir": "data/experiments/knn-explain-same-std/existing_experiment",
            "background_data": "data/files/train.csv",
            "explain_data": "data/files/test.csv",
        }

        new_file_dict = copy_data_files(temp_dir.name, old_file_dict, rsmtool_test_dir)

        source = "knn-explain-same-std"
        config_dict = {
            "description": "Explaning an KNeighborsRegressor model trained on all features.",
            "experiment_dir": new_file_dict["experiment_dir"],
            "experiment_id": "knn_same_std",
            "background_data": new_file_dict["background_data"],
            "background_kmeans_size": 50,
            "explain_data": new_file_dict["explain_data"],
            "standardize_features": False,
            "id_column": "ID",
            "sample_size": 10,
            "num_features_to_display": 15,
            "show_auto_cohorts": True,
        }

        # check `standardize_features` in the config is the same `False` as it was
        # before since that matches the value in rsmtool configuration
        with patch("rsmtool.rsmexplain.generate_report") as mock_generate_report:
            do_run_explain(source, config_dict)
            called_config = mock_generate_report.call_args[0][3]
            self.assertEqual(called_config["standardize_features"], False)

    def test_run_rsmexplain_different_truncate_outliers_value(self):
        """Check that rsmtool truncate outliers value overrides rsmexplain value."""
        # set up a temporary directory since we will be using getcwd
        temp_dir = tempfile.TemporaryDirectory(prefix=getcwd())

        old_file_dict = {
            "experiment_dir": "data/experiments/knn-explain-diff-trunc/existing_experiment",
            "background_data": "data/files/train.csv",
            "explain_data": "data/files/test.csv",
        }

        new_file_dict = copy_data_files(temp_dir.name, old_file_dict, rsmtool_test_dir)

        source = "knn-explain-diff-trunc"
        config_dict = {
            "description": "Explaning an KNeighborsRegressor model trained on all features.",
            "experiment_dir": new_file_dict["experiment_dir"],
            "experiment_id": "knn_diff_trunc",
            "background_data": new_file_dict["background_data"],
            "background_kmeans_size": 50,
            "explain_data": new_file_dict["explain_data"],
            "truncate_outliers": False,
            "id_column": "ID",
            "sample_size": 10,
            "num_features_to_display": 15,
            "show_auto_cohorts": True,
        }

        # check `truncate_outliers` in the config has been overridden to `True`
        # since that was the value in rsmtool configuration
        with patch("rsmtool.rsmexplain.generate_report") as mock_generate_report:
            do_run_explain(source, config_dict)
            called_config = mock_generate_report.call_args[0][3]
            self.assertEqual(called_config["truncate_outliers"], True)

    def test_run_rsmexplain_same_truncate_outliers_value(self):
        """Check that rsmexplain truncate outliers value does not change if matching rsmtool."""
        # set up a temporary directory since we will be using getcwd
        temp_dir = tempfile.TemporaryDirectory(prefix=getcwd())

        old_file_dict = {
            "experiment_dir": "data/experiments/knn-explain-same-trunc/existing_experiment",
            "background_data": "data/files/train.csv",
            "explain_data": "data/files/test.csv",
        }

        new_file_dict = copy_data_files(temp_dir.name, old_file_dict, rsmtool_test_dir)

        source = "knn-explain-same-trunc"
        config_dict = {
            "description": "Explaning an KNeighborsRegressor model trained on all features.",
            "experiment_dir": new_file_dict["experiment_dir"],
            "experiment_id": "knn_same_trunc",
            "background_data": new_file_dict["background_data"],
            "background_kmeans_size": 50,
            "explain_data": new_file_dict["explain_data"],
            "truncate_outliers": False,
            "id_column": "ID",
            "sample_size": 10,
            "num_features_to_display": 15,
            "show_auto_cohorts": True,
        }

        # check `truncate_outliers` in the config is the same `False` as it was
        # before since that matches the value in rsmtool configuration
        with patch("rsmtool.rsmexplain.generate_report") as mock_generate_report:
            do_run_explain(source, config_dict)
            called_config = mock_generate_report.call_args[0][3]
            self.assertEqual(called_config["truncate_outliers"], False)
