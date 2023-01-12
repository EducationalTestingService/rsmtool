import os
import tempfile
from os import getcwd
from os.path import join

from nose.tools import raises
from parameterized import param, parameterized

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


@parameterized(
    [
        param("lr-self-compare", "lr_subgroups_vs_lr_subgroups"),
        param("lr-different-compare", "lr_vs_lr_subset_features"),
        param("lr-self-compare-with-h2", "lr_with_h2_vs_lr_with_h2"),
        param("lr-self-compare-with-custom-order", "lr_subgroups_vs_lr_subgroups"),
        param("lr-self-compare-with-chosen-sections", "lr_subgroups_vs_lr_subgroups"),
        param(
            "lr-self-compare-with-custom-sections-and-custom-order",
            "lr_subgroups_vs_lr_subgroups",
        ),
        param("lr-self-compare-with-thumbnails", "lr_subgroups_vs_lr_subgroups"),
        param("linearsvr-self-compare", "LinearSVR_vs_LinearSVR"),
        param("lr-eval-self-compare", "lr_eval_with_h2_vs_lr_eval_with_h2"),
        param("lr-eval-tool-compare", "lr_with_h2_vs_lr_eval_with_h2"),
        param(
            "lr-self-compare-different-format",
            "lr_with_tsv_output_vs_lr_with_tsv_output",
        ),
        param(
            "lr-self-compare-with-subgroups-and-h2",
            "lr-subgroups-with-h2_vs_lr-subgroups-with-h2",
        ),
        param(
            "lr-self-compare-with-subgroups-and-edge-cases",
            "lr-subgroups-with-edge-cases_vs_lr-subgroups-with-edge-cases",
        ),
    ]
)
def test_run_experiment_parameterized(*args, **kwargs):
    if TEST_DIR:
        kwargs["given_test_dir"] = TEST_DIR
    check_run_comparison(*args, **kwargs)


def test_run_experiment_lr_compare_with_object():
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


def test_run_experiment_lr_compare_with_dictionary():
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


@raises(ValueError)
def test_run_comparison_wrong_input_format():
    config_list = [("experiment_id", "AAAA"), ("train_file", "some_path")]
    with tempfile.TemporaryDirectory() as temp_dir:
        run_comparison(config_list, temp_dir)


@raises(FileNotFoundError)
def test_run_experiment_lr_compare_wrong_directory():

    # basic rsmcompare experiment comparing a LinearRegression
    # experiment to itself
    source = "lr-self-compare-wrong-directory"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmcompare.json")
    do_run_comparison(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_lr_compare_wrong_experiment_id():

    # basic rsmcompare experiment comparing a LinearRegression
    # experiment to itself
    source = "lr-self-compare-wrong-id"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmcompare.json")
    do_run_comparison(source, config_file)
