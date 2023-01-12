import os
import tempfile
import warnings
from os import getcwd
from os.path import join

from nose.tools import raises
from parameterized import param, parameterized

from rsmtool import run_experiment
from rsmtool.configuration_parser import Configuration
from rsmtool.test_utils import check_run_experiment, copy_data_files, do_run_experiment

# allow test directory to be set via an environment variable
# which is needed for package testing
TEST_DIR = os.environ.get("TESTDIR", None)
if TEST_DIR:
    rsmtool_test_dir = TEST_DIR
else:
    from rsmtool.test_utils import rsmtool_test_dir


@parameterized(
    [
        param("lr-no-standardization", "lr_no_standardization"),
        param("lr-exclude-test-flags", "lr_exclude_test_flags"),
        param("lr-exclude-train-and-test-flags", "lr_exclude_train_and_test_flags"),
        param("lr-with-sas", "lr_with_sas"),
        param("lr-with-xlsx-output", "lr_with_xlsx_output", file_format="xlsx"),
        param("lr-with-tsv-output", "lr_with_tsv_output", file_format="tsv"),
        param("lr-with-thumbnails", "lr_with_thumbnails"),
        param(
            "lr-with-thumbnails-subgroups",
            "lr_with_thumbnails_subgroups",
            subgroups=["L1"],
        ),
        param("lr-with-feature-list", "lr_with_feature_list"),
        param("lr-with-length-non-numeric", "lr_with_length_non_numeric"),
        param(
            "lr-with-feature-list-and-transformation",
            "lr_with_feature_list_and_transformation",
        ),
        param("lr-with-trim-tolerance", "lr_with_trim_tolerance"),
        param(
            "lr-subgroups-with-dictionary-threshold-and-empty-group",
            "lr_subgroups_with_dictionary_threshold_and_empty_group",
            subgroups=["L1", "QUESTION"],
        ),
        param(
            "lr-subgroups-with-numeric-threshold-and-empty-group",
            "lr_subgroups_with_numeric_threshold_and_empty_group",
            subgroups=["L1", "QUESTION"],
        ),
        param(
            "lr-subgroups-h2-long-feature-names",
            "lr_subgroups_h2_long_feature_names",
            subgroups=["L1", "QUESTION"],
            consistency=True,
        ),
        param(
            "lr-subgroups-with-h2-but-only-for-nonscoreable",
            "lr_subgroups_with_h2_but_only_for_nonscoreable",
            subgroups=["L1", "QUESTION"],
            consistency=True,
            suppress_warnings_for=[UserWarning],
        ),
    ]
)
def test_run_experiment_parameterized(*args, **kwargs):
    if TEST_DIR:
        kwargs["given_test_dir"] = TEST_DIR
    check_run_experiment(*args, **kwargs)


def test_run_experiment_lr_with_object_and_configdir():
    """Test rsmtool using a Configuration object and specified configdir."""
    source = "lr-object"
    experiment_id = "lr_object"

    configdir = join(rsmtool_test_dir, "data", "experiments", source)

    config_dict = {
        "train_file": "../../files/train.csv",
        "id_column": "ID",
        "use_scaled_predictions": True,
        "test_label_column": "score",
        "train_label_column": "score",
        "test_file": "../../files/test.csv",
        "trim_max": 6,
        "features": "features.csv",
        "trim_min": 1,
        "model": "LinearRegression",
        "experiment_id": "lr_object",
        "description": "Using all features with an LinearRegression model.",
    }

    config_obj = Configuration(config_dict, configdir=configdir)

    check_run_experiment(source, experiment_id, config_obj_or_dict=config_obj)


def test_run_experiment_lr_with_object_no_configdir():
    """Test rsmtool using a Configuration object and no specified configdir."""
    source = "lr-object-no-path"
    experiment_id = "lr_object_no_path"

    # set up a temporary directory since
    # we will be using getcwd
    old_file_dict = {
        "train": "data/files/train.csv",
        "test": "data/files/test.csv",
        "features": "data/experiments/lr-object-no-path/features.csv",
    }

    temp_dir = tempfile.TemporaryDirectory(prefix=getcwd())
    new_file_dict = copy_data_files(temp_dir.name, old_file_dict, rsmtool_test_dir)

    config_dict = {
        "train_file": new_file_dict["train"],
        "id_column": "ID",
        "use_scaled_predictions": True,
        "test_label_column": "score",
        "train_label_column": "score",
        "test_file": new_file_dict["test"],
        "trim_max": 6,
        "features": new_file_dict["features"],
        "trim_min": 1,
        "model": "LinearRegression",
        "experiment_id": "lr_object_no_path",
        "description": "Using all features with an LinearRegression model.",
    }

    config_obj = Configuration(config_dict)

    check_run_experiment(source, experiment_id, config_obj_or_dict=config_obj)


def test_run_experiment_lr_with_dictionary():
    # Passing a dictionary as input.
    source = "lr-dictionary"
    experiment_id = "lr_dictionary"

    old_file_dict = {
        "train": "data/files/train.csv",
        "test": "data/files/test.csv",
        "features": "data/experiments/lr-dictionary/features.csv",
    }

    temp_dir = tempfile.TemporaryDirectory(prefix=getcwd())
    new_file_dict = copy_data_files(temp_dir.name, old_file_dict, rsmtool_test_dir)

    config_dict = {
        "train_file": new_file_dict["train"],
        "id_column": "ID",
        "use_scaled_predictions": True,
        "test_label_column": "score",
        "train_label_column": "score",
        "test_file": new_file_dict["test"],
        "trim_max": 6,
        "features": new_file_dict["features"],
        "trim_min": 1,
        "model": "LinearRegression",
        "experiment_id": "lr_dictionary",
        "description": "Using all features with an LinearRegression model.",
    }

    check_run_experiment(source, experiment_id, config_obj_or_dict=config_dict)


@raises(AttributeError)
def test_run_experiment_lr_with_object_and_filepath():
    """Test for a rare use case where an old Configuration object is passed."""
    source = "lr-object"
    experiment_id = "lr_object"

    config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")

    config_dict = {
        "train_file": "../../files/train.csv",
        "id_column": "ID",
        "use_scaled_predictions": True,
        "test_label_column": "score",
        "train_label_column": "score",
        "test_file": "../../files/test.csv",
        "trim_max": 6,
        "features": "features.csv",
        "trim_min": 1,
        "model": "LinearRegression",
        "experiment_id": "lr_object",
        "description": "Using all features with an LinearRegression model.",
    }

    config_obj = Configuration(config_dict)

    # we catch the deprecation warning triggered by this line
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        config_obj.filepath = config_file
    # we have to explicitly remove configdir attribute
    # since it will always be assigned a value by the current code
    del config_obj.configdir

    check_run_experiment(source, experiment_id, config_obj_or_dict=config_obj)


@raises(ValueError)
def test_run_experiment_wrong_input_format():
    config_list = [("experiment_id", "AAAA"), ("train_file", "some_path")]
    with tempfile.TemporaryDirectory() as temp_dir:
        run_experiment(config_list, temp_dir)


@raises(ValueError)
def test_run_experiment_duplicate_feature_names():

    # rsmtool experiment with duplicate feature names
    source = "lr-with-duplicate-feature-names"
    experiment_id = "lr_with_duplicate_feature_names"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")
    do_run_experiment(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_feature_json():
    # basic experiment with a LinearRegression model but using
    # feature json file

    source = "lr-feature-json"
    experiment_id = "lr"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")

    # run this experiment but suppress the expected deprecation warnings
    do_run_experiment(
        source, experiment_id, config_file, suppress_warnings_for=[DeprecationWarning]
    )


@raises(FileNotFoundError)
def test_run_experiment_wrong_train_file_path():
    # basic experiment with the path in train_file field pointing to
    # a non-existing file
    source = "lr-wrong-path"
    experiment_id = "lr"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")
    do_run_experiment(source, experiment_id, config_file)


@raises(FileNotFoundError)
def test_run_experiment_wrong_feature_file_path():
    # basic experiment with the path in features field pointing to
    # a non-existing file
    source = "lr-wrong-path-features"
    experiment_id = "lr"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")
    do_run_experiment(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_length_column_and_feature_list():
    # experiment with feature as list instead of file name
    # and length included in feature list and as length column

    source = "lr-with-length-and-feature-list"
    experiment_id = "lr_with_length_and_feature"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")
    do_run_experiment(source, experiment_id, config_file)
