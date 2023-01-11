import csv
import os
import tempfile
from glob import glob
from os import getcwd
from os.path import basename, exists, join

import pandas as pd
from nose.tools import raises
from pandas.testing import assert_frame_equal
from parameterized import param, parameterized

from rsmtool import compute_and_save_predictions, fast_predict
from rsmtool.configuration_parser import Configuration
from rsmtool.modeler import Modeler
from rsmtool.test_utils import (
    check_file_output,
    check_generated_output,
    check_report,
    check_run_prediction,
    check_scaled_coefficients,
    copy_data_files,
    do_run_experiment,
    do_run_prediction,
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
        param("lr-predict"),
        param("lr-predict-with-score"),
        param("lr-predict-missing-values", excluded=True),
        param("lr-predict-with-subgroups"),
        param("lr-predict-with-candidate"),
        param("lr-predict-illegal-transformations", excluded=True),
        param("lr-predict-tsv-input-files"),
        param("lr-predict-xlsx-input-files"),
        param("lr-predict-jsonlines-input-files"),
        param("lr-predict-nested-jsonlines-input-files"),
        param("lr-predict-no-standardization"),
        param("lr-predict-with-tsv-output", file_format="tsv"),
        param("lr-predict-with-xlsx-output", file_format="xlsx"),
        param("logistic-regression-predict"),
        param("logistic-regression-predict-expected-scores"),
        param("svc-predict-expected-scores"),
        param("lr-predict-with-custom-tolerance"),
        param("lr-predict-no-tolerance"),
    ]
)
def test_run_experiment_parameterized(*args, **kwargs):
    if TEST_DIR:
        kwargs["given_test_dir"] = TEST_DIR
    check_run_prediction(*args, **kwargs)


# Check that both rsmtool and rsmpredict generate the same files
def test_run_experiment_lr_rsmtool_and_rsmpredict():
    source = "lr-rsmtool-rsmpredict"
    experiment_id = "lr_rsmtool_rsmpredict"
    rsmtool_config_file = join(
        rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json"
    )
    do_run_experiment(source, experiment_id, rsmtool_config_file)
    rsmpredict_config_file = join(
        rsmtool_test_dir, "data", "experiments", source, "rsmpredict.json"
    )
    do_run_prediction(source, rsmpredict_config_file)
    output_dir = join("test_outputs", source, "output")
    expected_output_dir = join(rsmtool_test_dir, "data", "experiments", source, "output")
    csv_files = glob(join(output_dir, "*.csv"))
    html_report = join("test_outputs", source, "report", f"{experiment_id}_report.html")

    # Check the results for  rsmtool
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_file_output, csv_file, expected_csv_file

    yield check_scaled_coefficients, output_dir, experiment_id
    yield check_generated_output, csv_files, experiment_id, "rsmtool"
    yield check_report, html_report, True, False

    # check that the rsmpredict generated the same results
    for csv_pair in [
        ("predictions.csv", f"{experiment_id}_pred_processed.csv"),
        (
            "preprocessed_features.csv",
            f"{experiment_id}_test_preprocessed_features.csv",
        ),
    ]:
        output_file = join(output_dir, csv_pair[0])
        expected_output_file = join(expected_output_dir, csv_pair[1])

        yield check_file_output, output_file, expected_output_file


def test_run_experiment_lr_predict_with_object():
    """Test rsmpredict using the Configuration object, rather than a file."""
    source = "lr-predict-object"

    configdir = join(rsmtool_test_dir, "data", "experiments", source)

    config_dict = {
        "id_column": "ID",
        "input_features_file": "../../files/test.csv",
        "experiment_dir": "existing_experiment",
        "experiment_id": "lr",
    }

    config_obj = Configuration(config_dict, context="rsmpredict", configdir=configdir)

    check_run_prediction(source, given_test_dir=rsmtool_test_dir, config_obj_or_dict=config_obj)


def test_run_experiment_lr_predict_with_dictionary():
    """Test rsmpredict using the dictionary object, rather than a file."""
    source = "lr-predict-dict"

    # set up a temporary directory since
    # we will be using getcwd
    temp_dir = tempfile.TemporaryDirectory(prefix=getcwd())

    old_file_dict = {
        "feature_file": "data/files/test.csv",
        "experiment_dir": "data/experiments/lr-predict-dict/existing_experiment",
    }

    new_file_dict = copy_data_files(temp_dir.name, old_file_dict, rsmtool_test_dir)

    config_dict = {
        "id_column": "ID",
        "input_features_file": new_file_dict["feature_file"],
        "experiment_dir": new_file_dict["experiment_dir"],
        "experiment_id": "lr",
    }

    check_run_prediction(source, given_test_dir=rsmtool_test_dir, config_obj_or_dict=config_dict)


@raises(ValueError)
def test_run_experiment_lr_predict_with_repeated_ids():

    # rsmpredict experiment with non-unique ids
    source = "lr-predict-with-repeated-ids"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmpredict.json")
    do_run_prediction(source, config_file)


@raises(ValueError)
def test_compute_predictions_wrong_input_format():
    config_list = [("experiment_id", "AAAA"), ("train_file", "some_path")]
    with tempfile.TemporaryDirectory() as temp_dir:
        compute_and_save_predictions(config_list, temp_dir)


@raises(FileNotFoundError)
def test_run_experiment_lr_predict_missing_model_file():
    """Run rsmpredict experiment with missing model file."""
    source = "lr-predict-missing-model-file"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmpredict.json")
    do_run_prediction(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_lr_predict_missing_feature_file():
    """Run rsmpredict experiment with missing feature file."""
    source = "lr-predict-missing-feature-file"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmpredict.json")
    do_run_prediction(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_lr_predict_missing_postprocessing_file():
    """Run rsmpredict experiment with missing post-processing file."""
    source = "lr-predict-missing-postprocessing-file"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmpredict.json")
    do_run_prediction(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_predict_no_input_feature_file():
    """Run rsmpredict experiment with missing feature file."""
    source = "lr-predict-no-input-feature-file"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmpredict.json")
    do_run_prediction(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_predict_no_experiment_dir():
    """Run rsmpredict experiment with missing experiment dir."""
    source = "lr-predict-no-experiment-dir"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmpredict.json")
    do_run_prediction(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_predict_no_output_dir():
    """Run rsmpredict experiment with a missing "output" directory."""
    source = "lr-predict-no-output-dir"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmpredict.json")
    do_run_prediction(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_predict_no_experiment_id():
    """Run rsmpredict experiment with no experiment ID."""
    source = "lr-predict-no-experiment-id"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmpredict.json")
    do_run_prediction(source, config_file)


@raises(KeyError)
def test_run_experiment_lr_predict_missing_columns():
    """Run rsmpredict experiment with missing columns from the config file."""
    source = "lr-predict-missing-columns"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmpredict.json")
    do_run_prediction(source, config_file)


@raises(KeyError)
def test_run_experiment_lr_predict_missing_feature():
    """Run rsmpredict experiment with missing features."""
    source = "lr-predict-missing-feature"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmpredict.json")
    do_run_prediction(source, config_file)


@raises(ValueError)
def test_run_experiment_lr_predict_no_numeric_feature_values():
    """Run rsmpredict experiment with missing post-processing file."""
    source = "lr-predict-no-numeric-feature-values"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmpredict.json")
    do_run_prediction(source, config_file)


@raises(ValueError)
def test_run_experiment_predict_expected_scores_builtin_model():
    """Run rsmpredict experiment for expected scores with unsupported built-in model."""
    source = "lr-predict-expected-scores-builtin-model"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmpredict.json")
    do_run_prediction(source, config_file)


@raises(ValueError)
def test_run_experiment_predict_expected_scores_wrong_skll_model():
    """Run rsmpredict experiment for expected scores with an unsupported SKLL learner."""
    source = "predict-expected-scores-wrong-skll-model"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmpredict.json")
    do_run_prediction(source, config_file)


@raises(ValueError)
def test_run_experiment_predict_expected_scores_non_probablistic_svc():
    """Run rsmpredict experiment for expected scores with a non-probabilistic learner."""
    source = "predict-expected-scores-non-probabilistic-svc"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmpredict.json")
    do_run_prediction(source, config_file)


def check_fast_predict(source, do_trim=False, do_scale=False):
    """Ensure that predictions from `fast_predict()` match expected predictions."""
    # define the paths to the various files we need
    test_file = join(rsmtool_test_dir, "data", "files", "test.csv")
    existing_experiment_dir = join(
        rsmtool_test_dir, "data", "experiments", source, "existing_experiment", "output"
    )
    feature_info_file = join(existing_experiment_dir, "lr_feature.csv")
    postprocessing_params_file = join(existing_experiment_dir, "lr_postprocessing_params.csv")
    model_file = join(existing_experiment_dir, "lr.model")

    # read in the files
    df_test = pd.read_csv(test_file, usecols=[f"FEATURE{i}" for i in range(1, 9)])
    df_feature_info = pd.read_csv(feature_info_file, index_col=0)
    with open(postprocessing_params_file, "r") as paramfh:
        reader = csv.DictReader(paramfh)
        params = next(reader)

    # initialize the modeler instance
    modeler = Modeler.load_from_file(model_file)

    # initialize the keyword arguments we want to passs
    kwargs = {}
    if do_trim:
        kwargs["trim_min"] = int(params["trim_min"])
        kwargs["trim_max"] = int(params["trim_max"])
        if "trim_tolerance" in params:
            kwargs["trim_tolerance"] = float(params["trim_tolerance"])

    if do_scale:
        kwargs["train_predictions_mean"] = float(params["train_predictions_mean"])
        kwargs["train_predictions_sd"] = float(params["train_predictions_sd"])
        kwargs["h1_mean"] = float(params["h1_mean"])
        kwargs["h1_sd"] = float(params["h1_sd"])

    # initialize a variable to hold all the predictions
    prediction_dicts = []
    for input_features in df_test.to_dict(orient="records"):
        tolerance = params.get("trim_tolerance")
        tolerance = float(tolerance) if tolerance else tolerance
        predictions = fast_predict(input_features, modeler, df_feature_info, **kwargs)
        prediction_dicts.append(predictions)

    # combine all the computed predictions into a data frame
    df_computed_predictions = pd.DataFrame(prediction_dicts)

    # read in the expected predictions
    df_expected_predictions = pd.read_csv(
        join(rsmtool_test_dir, "data", "experiments", source, "output", "predictions.csv")
    )
    df_expected_predictions = df_expected_predictions.drop("spkitemid", axis="columns")

    # keep only the needed columns for each test
    if not do_trim:
        df_expected_predictions = df_expected_predictions.drop(
            ["raw_trim", "scale_trim", "raw_trim_round", "scale_trim_round"],
            axis="columns",
            errors="ignore",
        )

    if not do_scale:
        df_expected_predictions = df_expected_predictions.drop(
            ["scale", "scale_trim", "scale_trim_round"], axis="columns", errors="ignore"
        )

    # check that the predictions are equal
    assert_frame_equal(
        df_computed_predictions.sort_index(axis="columns"),
        df_expected_predictions.sort_index(axis="columns"),
    )


def test_fast_predict():  # noqa: D103
    yield check_fast_predict, "lr-predict",
    yield check_fast_predict, "lr-predict", True, False
    yield check_fast_predict, "lr-predict", False, True
    yield check_fast_predict, "lr-predict", True, True
    yield check_fast_predict, "lr-predict-no-tolerance", True, True


@raises(ValueError)
def test_fast_predict_non_numeric():
    """Check that ``fast_predict()`` raises an error with non-numeric values."""
    existing_experiment_dir = join(
        rsmtool_test_dir,
        "data",
        "experiments",
        "lr-predict",
        "existing_experiment",
        "output",
    )
    feature_info_file = join(existing_experiment_dir, "lr_feature.csv")
    model_file = join(existing_experiment_dir, "lr.model")

    # read in the files
    df_feature_info = pd.read_csv(feature_info_file, index_col=0)

    # initialize the modeler instance
    modeler = Modeler.load_from_file(model_file)

    # input features with non-numeric value
    input_features = {
        "FEATURE1": 6.0,
        "FEATURE2": "foobar",
        "FEATURE3": -0.2,
        "FEATURE4": 0,
        "FEATURE5": -0.1,
        "FEATURE6": 5.0,
        "FEATURE7": 12,
        "FEATURE8": -7000,
    }
    _ = fast_predict(input_features, modeler, df_feature_info)
