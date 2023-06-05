import json
import os
from os.path import join

from nose.tools import assert_equal, raises
from parameterized import param, parameterized
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


@parameterized(
    [
        param("lr-with-h2-include-zeros", "lr_with_h2_include_zeros", consistency=True),
        param("lr-with-h2-and-length", "lr_with_h2_and_length", consistency=True),
        param("lr-with-h2-named-sc1", "lr_with_h2_named_sc1", consistency=True),
        param("lars", "Lars", skll=True),
        param("lars-custom-objective", "Lars_custom_objective", skll=True),
        param("logistic-regression", "LogisticRegression", skll=True),
        param(
            "logistic-regression-custom-objective",
            "LogisticRegression_custom_objective",
            skll=True,
        ),
        param(
            "logistic-regression-custom-objective-and-params",
            "LogisticRegression_custom_objective_and_params",
            skll=True,
        ),
        param(
            "logistic-regression-expected-scores",
            "LogisticRegression_expected_scores",
            skll=True,
        ),
        param("svc", "SVC", skll=True),
        param("svc-custom-objective", "SVC_custom_objective", skll=True),
        param(
            "svc-custom-objective-and-params",
            "SVC_custom_objective_and_params",
            skll=True,
        ),
        param("svc-expected-scores", "SVC_expected_scores", skll=True),
        param("dummyregressor", "DummyRegressor", skll=True),
        param(
            "dummyregressor-custom-objective",
            "DummyRegressor_custom_objective",
            skll=True,
        ),
        param("ridge", "Ridge", skll=True),
        param("ridge-custom-objective", "Ridge_custom_objective", skll=True),
        param("ridge-custom-params", "Ridge_custom_params", skll=True),
        param("linearsvr", "LinearSVR", skll=True),
        param("linearsvr-custom-objective", "LinearSVR_custom_objective", skll=True),
        param(
            "wls", "wls", skll=True
        ),  # treat this as SKLL since we don't want to test coefficients
        param("rebalancedlr", "rebalancedlr"),
        param("lassofixedlambdathenlr", "lassofixedlambdathenlr"),
        param("positivelassocvthenlr", "positivelassocvthenlr"),
        param("nnlr", "nnlr"),
        param("nnlr_iterative", "nnlr_iterative"),
        param("lassofixedlambdathennnlr", "lassofixedlambdathennnlr"),
        param("lassofixedlambda", "lassofixedlambda"),
        param("positivelassocv", "positivelassocv"),
        param("equalweightslr", "equalweightslr"),
        param("lr-with-length-string", "lr_with_length_string"),
    ]
)
def test_run_experiment_parameterized(*args, **kwargs):
    if TEST_DIR:
        kwargs["given_test_dir"] = TEST_DIR

    # suppress known convergence warnings for LinearSVR-based experiments
    # TODO: once SKLL hyperparameters can be passed, replace this code
    if args[0].startswith("linearsvr"):
        kwargs["suppress_warnings_for"] = [ConvergenceWarning]

    check_run_experiment(*args, **kwargs)


@raises(ValueError)
def test_run_experiment_empwtdropneg():
    # rsmtool experiment with no longer supported empWtDropNeg model
    source = "empwtdropneg"
    experiment_id = "empWtDropNeg"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")
    do_run_experiment(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_requested_feature_zero_sd():
    # rsmtool experiment when a requested feature has zero sd
    source = "lr-with-requested-feature-with-zero-sd"
    experiment_id = "lr_with_requested_feature_with_zero_sd"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")
    do_run_experiment(source, experiment_id, config_file)


def test_run_experiment_with_warnings():
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

    assert_equal(len(syntax_warnings), 1)
    assert_equal(len(deprecation_warnings), 2)
    assert_equal(len(unicode_warnings), 1)
    assert_equal(len(runtime_warnings), 1)
    assert_equal(len(user_warnings), 1)


@raises(ValueError)
def test_same_id_linear_then_non_linear_raises_error():
    experiment_path = join(rsmtool_test_dir, "data", "experiments", "lr")
    configpath = join(experiment_path, "lr.json")
    configdict = json.load(open(configpath, "r"))

    output_dir = "test_outputs/same-id-different-model"
    config = Configuration(configdict, configdir=experiment_path)
    run_experiment(config, output_dir, overwrite_output=True)

    config["model"] = "SVC"
    run_experiment(config, output_dir, overwrite_output=True)
