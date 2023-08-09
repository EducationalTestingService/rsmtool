import csv
import itertools
import os
import tempfile
import unittest
from glob import glob
from os import getcwd
from os.path import basename, exists, join

import pandas as pd
from nose2.tools import params
from pandas.testing import assert_frame_equal

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


class TestExperimentRsmpredict(unittest.TestCase):
    @params(
        {"source": "lr-predict"},
        {"source": "lr-predict-with-score"},
        {"source": "lr-predict-missing-values", "excluded": "True"},
        {"source": "lr-predict-with-subgroups"},
        {"source": "lr-predict-with-candidate"},
        {"source": "lr-predict-illegal-transformations", "excluded": "True"},
        {"source": "lr-predict-tsv-input-files"},
        {"source": "lr-predict-xlsx-input-files"},
        {"source": "lr-predict-jsonlines-input-files"},
        {"source": "lr-predict-nested-jsonlines-input-files"},
        {"source": "lr-predict-no-standardization"},
        {"source": "lr-predict-with-tsv-output", "file_format": "tsv"},
        {"source": "lr-predict-with-xlsx-output", "file_format": "xlsx"},
        {"source": "logistic-regression-predict"},
        {"source": "logistic-regression-predict-expected-scores"},
        {"source": "svc-predict-expected-scores"},
        {"source": "lr-predict-with-custom-tolerance"},
        {"source": "lr-predict-no-tolerance"},
    )
    def test_run_experiment_parameterized(self, kwargs):
        if TEST_DIR:
            kwargs["given_test_dir"] = TEST_DIR
        check_run_prediction(**kwargs)

    # Check that both rsmtool and rsmpredict generate the same files
    def test_run_experiment_lr_rsmtool_and_rsmpredict(self):
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

    def test_run_experiment_lr_predict_with_object(self):
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

    def test_run_experiment_lr_predict_with_dictionary(self):
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

        check_run_prediction(
            source, given_test_dir=rsmtool_test_dir, config_obj_or_dict=config_dict
        )

    def test_run_experiment_lr_predict_with_repeated_ids(self):
        # rsmpredict experiment with non-unique ids
        source = "lr-predict-with-repeated-ids"
        config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmpredict.json")
        with self.assertRaises(ValueError):
            do_run_prediction(source, config_file)

    def test_compute_predictions_wrong_input_format(self):
        config_list = [("experiment_id", "AAAA"), ("train_file", "some_path")]
        with self.assertRaises(ValueError):
            with tempfile.TemporaryDirectory() as temp_dir:
                compute_and_save_predictions(config_list, temp_dir)

    def test_run_experiment_lr_predict_missing_model_file(self):
        """Run rsmpredict experiment with missing model file."""
        source = "lr-predict-missing-model-file"
        config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmpredict.json")
        with self.assertRaises(FileNotFoundError):
            do_run_prediction(source, config_file)

    def test_run_experiment_lr_predict_missing_feature_file(self):
        """Run rsmpredict experiment with missing feature file."""
        source = "lr-predict-missing-feature-file"
        config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmpredict.json")
        with self.assertRaises(FileNotFoundError):
            do_run_prediction(source, config_file)

    def test_run_experiment_lr_predict_missing_postprocessing_file(self):
        """Run rsmpredict experiment with missing post-processing file."""
        source = "lr-predict-missing-postprocessing-file"
        config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmpredict.json")
        with self.assertRaises(FileNotFoundError):
            do_run_prediction(source, config_file)

    def test_run_experiment_predict_no_input_feature_file(self):
        """Run rsmpredict experiment with missing feature file."""
        source = "lr-predict-no-input-feature-file"
        config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmpredict.json")
        with self.assertRaises(FileNotFoundError):
            do_run_prediction(source, config_file)

    def test_run_experiment_predict_no_experiment_dir(self):
        """Run rsmpredict experiment with missing experiment dir."""
        source = "lr-predict-no-experiment-dir"
        config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmpredict.json")
        with self.assertRaises(FileNotFoundError):
            do_run_prediction(source, config_file)

    def test_run_experiment_predict_no_output_dir(self):
        """Run rsmpredict experiment with a missing "output" directory."""
        source = "lr-predict-no-output-dir"
        config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmpredict.json")
        with self.assertRaises(FileNotFoundError):
            do_run_prediction(source, config_file)

    def test_run_experiment_predict_no_experiment_id(self):
        """Run rsmpredict experiment with no experiment ID."""
        source = "lr-predict-no-experiment-id"
        config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmpredict.json")
        with self.assertRaises(FileNotFoundError):
            do_run_prediction(source, config_file)

    def test_run_experiment_lr_predict_missing_columns(self):
        """Run rsmpredict experiment with missing columns from the config file."""
        source = "lr-predict-missing-columns"
        config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmpredict.json")
        with self.assertRaises(KeyError):
            do_run_prediction(source, config_file)

    def test_run_experiment_lr_predict_missing_feature(self):
        """Run rsmpredict experiment with missing features."""
        source = "lr-predict-missing-feature"
        config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmpredict.json")
        with self.assertRaises(KeyError):
            do_run_prediction(source, config_file)

    def test_run_experiment_lr_predict_no_numeric_feature_values(self):
        """Run rsmpredict experiment with missing post-processing file."""
        source = "lr-predict-no-numeric-feature-values"
        config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmpredict.json")
        with self.assertRaises(ValueError):
            do_run_prediction(source, config_file)

    def test_run_experiment_predict_expected_scores_builtin_model(self):
        """Run rsmpredict experiment for expected scores with unsupported built-in model."""
        source = "lr-predict-expected-scores-builtin-model"
        config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmpredict.json")
        with self.assertRaises(ValueError):
            do_run_prediction(source, config_file)

    def test_run_experiment_predict_expected_scores_wrong_skll_model(self):
        """Run rsmpredict experiment for expected scores with an unsupported SKLL learner."""
        source = "predict-expected-scores-wrong-skll-model"
        config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmpredict.json")
        with self.assertRaises(ValueError):
            do_run_prediction(source, config_file)

    def test_run_experiment_predict_expected_scores_non_probablistic_svc(self):
        """Run rsmpredict experiment for expected scores with a non-probabilistic learner."""
        source = "predict-expected-scores-non-probabilistic-svc"
        config_file = join(rsmtool_test_dir, "data", "experiments", source, "rsmpredict.json")
        with self.assertRaises(ValueError):
            do_run_prediction(source, config_file)

    def check_fast_predict(
        self,
        source,
        do_trim=False,
        do_scale=False,
        explicit_trim_scale_args=False,
        use_default_trim_tolerance=False,
    ):
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

        # initialize the keyword arguments we want to pass
        kwargs = {}
        if explicit_trim_scale_args:
            kwargs["df_feature_info"] = df_feature_info
        if do_trim:
            kwargs["trim"] = True
            if explicit_trim_scale_args:
                kwargs["trim_min"] = int(params["trim_min"])
                kwargs["trim_max"] = int(params["trim_max"])
                if use_default_trim_tolerance:
                    delattr(modeler, "trim_tolerance")
                elif "trim_tolerance" in params:
                    kwargs["trim_tolerance"] = float(params["trim_tolerance"])

        if do_scale:
            kwargs["scale"] = True
            if explicit_trim_scale_args:
                kwargs["train_predictions_mean"] = float(params["train_predictions_mean"])
                kwargs["train_predictions_sd"] = float(params["train_predictions_sd"])
                kwargs["h1_mean"] = float(params["h1_mean"])
                kwargs["h1_sd"] = float(params["h1_sd"])

        # initialize a variable to hold all the predictions
        prediction_dicts = []
        for input_features in df_test.to_dict(orient="records"):
            predictions = fast_predict(input_features, modeler, **kwargs)
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

    def test_fast_predict(self):  # noqa: D103
        for do_trim, do_scale, explicit_trim_scale_args in itertools.product(
            [True, False], repeat=3
        ):
            for source in ["lr-predict", "lr-predict-no-tolerance"]:
                yield self.check_fast_predict, source, do_trim, do_scale, explicit_trim_scale_args

        # Use the default trim tolerance value
        for explicit_trim_scale_args in [True, False]:
            yield self.check_fast_predict, "lr-predict", True, False, explicit_trim_scale_args, True

    def test_fast_predict_non_numeric(self):
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
        with self.assertRaises(ValueError):
            _ = fast_predict(input_features, modeler, df_feature_info)

    def test_fast_predict_missing_feature_info(self):
        """Check case where feature information is missing."""
        existing_experiment_dir = join(
            rsmtool_test_dir,
            "data",
            "experiments",
            "lr-predict",
            "existing_experiment",
            "output",
        )
        model_file = join(existing_experiment_dir, "lr.model")

        # initialize the modeler instance and remove the
        # ``feature_info`` attribute
        modeler = Modeler.load_from_file(model_file)
        delattr(modeler, "feature_info")

        input_features = {
            "FEATURE1": 6.0,
            "FEATURE2": 1.0,
            "FEATURE3": -0.2,
            "FEATURE4": 0,
            "FEATURE5": -0.1,
            "FEATURE6": 5.0,
            "FEATURE7": 12,
            "FEATURE8": -7000,
        }
        with self.assertRaises(ValueError):
            _ = fast_predict(input_features, modeler)

    def test_fast_predict_feature_info_none(self):
        """Check case where feature information is ``None``."""
        existing_experiment_dir = join(
            rsmtool_test_dir,
            "data",
            "experiments",
            "lr-predict",
            "existing_experiment",
            "output",
        )
        model_file = join(existing_experiment_dir, "lr.model")

        # initialize the modeler instance with a bare learner, in which
        # case no feature_info attribute will exist
        modeler = Modeler.load_from_learner(Modeler.load_from_file(model_file).learner)

        input_features = {
            "FEATURE1": 6.0,
            "FEATURE2": 1.0,
            "FEATURE3": -0.2,
            "FEATURE4": 0,
            "FEATURE5": -0.1,
            "FEATURE6": 5.0,
            "FEATURE7": 12,
            "FEATURE8": -7000,
        }
        with self.assertRaises(ValueError):
            _ = fast_predict(input_features, modeler)

    def test_fast_predict_scaling_params_but_scale_false(self):
        """Check case when scaling is turned off yet scaling-related parameters are used."""
        existing_experiment_dir = join(
            rsmtool_test_dir,
            "data",
            "experiments",
            "lr-predict",
            "existing_experiment",
            "output",
        )
        model_file = join(existing_experiment_dir, "lr.model")
        modeler = Modeler.load_from_file(model_file)
        input_features = {
            "FEATURE1": 6.0,
            "FEATURE2": 1.0,
            "FEATURE3": -0.2,
            "FEATURE4": 0,
            "FEATURE5": -0.1,
            "FEATURE6": 5.0,
            "FEATURE7": 12,
            "FEATURE8": -7000,
        }

        # For each scaling-related parameter, simply get the parameter
        # from the modeler and try to pass it in with scale=False
        for param in ["train_predictions_mean", "train_predictions_sd", "h1_mean", "h1_sd"]:
            scale_params = {param: getattr(modeler, param)}
            with self.assertRaises(ValueError):
                _ = fast_predict(
                    input_features,
                    modeler,
                    scale=False,
                    **scale_params,
                )

    def test_fast_predict_scale_true_missing_scale_params(self):
        """Check case when scaling is turned on yet scaling-related parameters are missing."""
        existing_experiment_dir = join(
            rsmtool_test_dir,
            "data",
            "experiments",
            "lr-predict",
            "existing_experiment",
            "output",
        )
        model_file = join(existing_experiment_dir, "lr.model")
        input_features = {
            "FEATURE1": 6.0,
            "FEATURE2": 1.0,
            "FEATURE3": -0.2,
            "FEATURE4": 0,
            "FEATURE5": -0.1,
            "FEATURE6": 5.0,
            "FEATURE7": 12,
            "FEATURE8": -7000,
        }

        # For each scaling-related parameter, simply delete the
        # parameter-related attribute from the modeler and then call
        # ``fast_predict`` with scale=True
        for param in ["train_predictions_mean", "train_predictions_sd", "h1_mean", "h1_sd"]:
            modeler = Modeler.load_from_file(model_file)
            delattr(modeler, param)
            with self.assertRaises(ValueError):
                _ = fast_predict(
                    input_features,
                    modeler,
                    scale=True,
                )

    def test_fast_predict_scale_true_scale_attributes_none(self):
        """Check case when scaling is turned on yet scaling-related parameters are ``None``."""
        # Load existing model, save its feature info attribute (for later
        # use), then reload from its learner attribute to mimic loading
        # from a bare SKLL learner
        existing_experiment_dir = join(
            rsmtool_test_dir,
            "data",
            "experiments",
            "lr-predict",
            "existing_experiment",
            "output",
        )
        modeler = Modeler.load_from_file(join(existing_experiment_dir, "lr.model"))
        feature_info = modeler.feature_info
        input_features = {
            "FEATURE1": 6.0,
            "FEATURE2": 1.0,
            "FEATURE3": -0.2,
            "FEATURE4": 0,
            "FEATURE5": -0.1,
            "FEATURE6": 5.0,
            "FEATURE7": 12,
            "FEATURE8": -7000,
        }
        modeler = Modeler.load_from_learner(modeler.learner)

        # Make prediction with scaling even though scale-related
        # attributes will have None values
        with self.assertRaises(ValueError):
            _ = fast_predict(
                input_features,
                modeler,
                df_feature_info=feature_info,
                scale=True,
            )

    def test_fast_predict_trimming_params_but_trim_false(self):
        """Check case when trimming is turned off yet trimming-related parameters are used."""
        existing_experiment_dir = join(
            rsmtool_test_dir,
            "data",
            "experiments",
            "lr-predict",
            "existing_experiment",
            "output",
        )
        model_file = join(existing_experiment_dir, "lr.model")
        modeler = Modeler.load_from_file(model_file)
        input_features = {
            "FEATURE1": 6.0,
            "FEATURE2": 1.0,
            "FEATURE3": -0.2,
            "FEATURE4": 0,
            "FEATURE5": -0.1,
            "FEATURE6": 5.0,
            "FEATURE7": 12,
            "FEATURE8": -7000,
        }

        # For each trimming-related parameter, simply get the parameter
        # from the modeler and try to pass it in with trim=False
        for param in ["trim_min", "trim_max", "trim_tolerance"]:
            trim_params = {param: getattr(modeler, param)}
            with self.assertRaises(ValueError):
                _ = fast_predict(
                    input_features,
                    modeler,
                    trim=False,
                    **trim_params,
                )

    def test_fast_predict_trim_true_missing_trim_params(self):
        """Check case when trimming is turned on yet trimming-related parameters are missing."""
        existing_experiment_dir = join(
            rsmtool_test_dir,
            "data",
            "experiments",
            "lr-predict",
            "existing_experiment",
            "output",
        )
        model_file = join(existing_experiment_dir, "lr.model")
        input_features = {
            "FEATURE1": 6.0,
            "FEATURE2": 1.0,
            "FEATURE3": -0.2,
            "FEATURE4": 0,
            "FEATURE5": -0.1,
            "FEATURE6": 5.0,
            "FEATURE7": 12,
            "FEATURE8": -7000,
        }

        # For each trimming-related parameter, simply delete the
        # parameter-related attribute from the modeler and then call
        # ``fast_predict`` with trim=True
        for param in ["trim_min", "trim_max"]:
            modeler = Modeler.load_from_file(model_file)
            delattr(modeler, param)
            with self.assertRaises(ValueError):
                _ = fast_predict(
                    input_features,
                    modeler,
                    trim=True,
                )

    def test_fast_predict_trim_true_trim_attributes_none(self):
        """Check case when trimming is turned on yet trimming-related parameters are ``None``."""
        # Load existing model, save its feature info attribute (for later
        # use), then reload from its learner attribute to mimic loading
        # from a bare SKLL learner
        existing_experiment_dir = join(
            rsmtool_test_dir,
            "data",
            "experiments",
            "lr-predict",
            "existing_experiment",
            "output",
        )
        modeler = Modeler.load_from_file(join(existing_experiment_dir, "lr.model"))
        feature_info = modeler.feature_info
        input_features = {
            "FEATURE1": 6.0,
            "FEATURE2": 1.0,
            "FEATURE3": -0.2,
            "FEATURE4": 0,
            "FEATURE5": -0.1,
            "FEATURE6": 5.0,
            "FEATURE7": 12,
            "FEATURE8": -7000,
        }
        modeler = Modeler.load_from_learner(modeler.learner)

        # Make prediction with trimming even though trim-related
        # attributes will have None values
        with self.assertRaises(ValueError):
            _ = fast_predict(
                input_features,
                modeler,
                df_feature_info=feature_info,
                trim=True,
            )
