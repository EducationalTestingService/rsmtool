import json
import os
import unittest
from os.path import join
from unittest.mock import Mock, patch

from nose2.tools import params
from sklearn.exceptions import ConvergenceWarning

from rsmtool import run_experiment
from rsmtool.configuration_parser import Configuration
from rsmtool.test_utils import (
    check_run_experiment,
    collect_warning_messages_from_report,
    do_run_experiment,
)

# allow test directory to be set via an environment variable
# which is needed for package testing
TEST_DIR = os.environ.get("TESTDIR", None)
if TEST_DIR:
    rsmtool_test_dir = TEST_DIR
else:
    from rsmtool.test_utils import rsmtool_test_dir


class TestExperimentRsmtool4(unittest.TestCase):
    @params(
        {
            "source": "lr-with-h2-include-zeros",
            "experiment_id": "lr_with_h2_include_zeros",
            "consistency": "True",
        },
        {
            "source": "lr-with-h2-and-length",
            "experiment_id": "lr_with_h2_and_length",
            "consistency": "True",
        },
        {
            "source": "lr-with-h2-named-sc1",
            "experiment_id": "lr_with_h2_named_sc1",
            "consistency": "True",
        },
        {"source": "lars", "experiment_id": "Lars", "skll": "True"},
        {
            "source": "lars-custom-objective",
            "experiment_id": "Lars_custom_objective",
            "skll": "True",
        },
        {"source": "logistic-regression", "experiment_id": "LogisticRegression", "skll": "True"},
        {
            "source": "logistic-regression-custom-objective",
            "experiment_id": "LogisticRegression_custom_objective",
            "skll": "True",
        },
        {
            "source": "logistic-regression-custom-objective-and-params",
            "experiment_id": "LogisticRegression_custom_objective_and_params",
            "skll": "True",
        },
        {
            "source": "logistic-regression-expected-scores",
            "experiment_id": "LogisticRegression_expected_scores",
            "skll": "True",
        },
        {"source": "svc", "experiment_id": "SVC", "skll": "True"},
        {"source": "svc-custom-objective", "experiment_id": "SVC_custom_objective", "skll": "True"},
        {
            "source": "svc-custom-objective-and-params",
            "experiment_id": "SVC_custom_objective_and_params",
            "skll": "True",
        },
        {"source": "svc-expected-scores", "experiment_id": "SVC_expected_scores", "skll": "True"},
        {"source": "dummyregressor", "experiment_id": "DummyRegressor", "skll": "True"},
        {
            "source": "dummyregressor-custom-objective",
            "experiment_id": "DummyRegressor_custom_objective",
            "skll": "True",
        },
        {"source": "ridge", "experiment_id": "Ridge", "skll": "True"},
        {
            "source": "ridge-custom-objective",
            "experiment_id": "Ridge_custom_objective",
            "skll": "True",
        },
        {"source": "ridge-custom-params", "experiment_id": "Ridge_custom_params", "skll": "True"},
        {"source": "linearsvr", "experiment_id": "LinearSVR", "skll": "True"},
        {
            "source": "linearsvr-custom-objective",
            "experiment_id": "LinearSVR_custom_objective",
            "skll": "True",
        },
        {"source": "wls", "experiment_id": "wls", "skll": "True"},
        {"source": "rebalancedlr", "experiment_id": "rebalancedlr"},
        {"source": "lassofixedlambdathenlr", "experiment_id": "lassofixedlambdathenlr"},
        {"source": "positivelassocvthenlr", "experiment_id": "positivelassocvthenlr"},
        {"source": "nnlr", "experiment_id": "nnlr"},
        {"source": "nnlr_iterative", "experiment_id": "nnlr_iterative"},
        {"source": "lassofixedlambdathennnlr", "experiment_id": "lassofixedlambdathennnlr"},
        {"source": "lassofixedlambda", "experiment_id": "lassofixedlambda"},
        {"source": "positivelassocv", "experiment_id": "positivelassocv"},
        {"source": "equalweightslr", "experiment_id": "equalweightslr"},
        {"source": "lr-with-length-string", "experiment_id": "lr_with_length_string"},
    )
    def test_run_experiment_parameterized(self, kwargs):
        if TEST_DIR:
            kwargs["given_test_dir"] = TEST_DIR

        # suppress known convergence warnings for LinearSVR-based experiments
        # TODO: once SKLL hyperparameters can be passed, replace this code
        if kwargs["source"].startswith("linearsvr"):
            kwargs["suppress_warnings_for"] = [ConvergenceWarning]

        check_run_experiment(**kwargs)

    def test_run_experiment_empwtdropneg(self):
        # rsmtool experiment with no longer supported empWtDropNeg model
        source = "empwtdropneg"
        experiment_id = "empWtDropNeg"
        config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")
        with self.assertRaises(ValueError):
            do_run_experiment(source, experiment_id, config_file)

    def test_run_experiment_requested_feature_zero_sd(self):
        # rsmtool experiment when a requested feature has zero sd
        source = "lr-with-requested-feature-with-zero-sd"
        experiment_id = "lr_with_requested_feature_with_zero_sd"
        config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")
        with self.assertRaises(ValueError):
            do_run_experiment(source, experiment_id, config_file)

    def test_run_experiment_with_warnings(self):
        source = "lr-with-warnings"
        experiment_id = "lr_with_warnings"
        config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")

        do_run_experiment(source, experiment_id, config_file)

        html_file = join("test_outputs", source, "report", experiment_id + "_report.html")
        report_warnings = collect_warning_messages_from_report(html_file)

        syntax_warnings = [msg for msg in report_warnings if "syntax warning" in msg]
        deprecation_warnings = [msg for msg in report_warnings if "deprecation warning" in msg]
        unicode_warnings = [msg for msg in report_warnings if "unicode warning" in msg]
        runtime_warnings = [msg for msg in report_warnings if "runtime warning" in msg]
        user_warnings = [msg for msg in report_warnings if "user warning" in msg]

        self.assertEqual(len(syntax_warnings), 1)
        self.assertEqual(len(deprecation_warnings), 2)
        self.assertEqual(len(unicode_warnings), 1)
        self.assertEqual(len(runtime_warnings), 1)
        self.assertEqual(len(user_warnings), 1)

    def test_same_id_linear_then_non_linear_raises_error(self):
        experiment_path = join(rsmtool_test_dir, "data", "experiments", "lr")
        configpath = join(experiment_path, "lr.json")
        configdict = json.load(open(configpath, "r"))

        output_dir = "test_outputs/same-id-different-model"
        config = Configuration(configdict, configdir=experiment_path)
        run_experiment(config, output_dir, overwrite_output=True)

        config["model"] = "SVC"
        with self.assertRaises(ValueError):
            run_experiment(config, output_dir, overwrite_output=True)

    @patch("wandb.init")
    @patch("wandb.plot.confusion_matrix")
    def test_run_experiment_with_wandb(self, mock_plot_conf_mat, mock_wandb_init):
        source = "wandb"
        experiment_id = "wandb"
        config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")
        mock_wandb_run = Mock()
        mock_wandb_init.return_value = mock_wandb_run
        do_run_experiment(source, experiment_id, config_file)
        mock_wandb_init.assert_called_with(project="wandb_project", entity="wandb_entity")
        mock_wandb_run.log_artifact.assert_called_once()
        mock_plot_conf_mat.assert_called()
