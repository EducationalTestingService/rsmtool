import argparse
import filecmp
import warnings
from io import StringIO
from itertools import count, product
from os import environ, getcwd, listdir, makedirs, unlink
from os.path import abspath, join, relpath
from pathlib import Path
from shutil import rmtree
from tempfile import NamedTemporaryFile, TemporaryDirectory, mkdtemp
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pandas as pd
from nose.tools import assert_dict_equal, assert_equal, eq_, ok_, raises
from numpy.testing import assert_almost_equal, assert_array_equal
from pandas.testing import assert_frame_equal
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import CompleteStyle
from sklearn.datasets import make_classification
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import cohen_kappa_score
from skll.data import FeatureSet
from skll.learner import Learner
from skll.metrics import kappa

from rsmtool.configuration_parser import Configuration, ConfigurationParser
from rsmtool.reader import DataReader
from rsmtool.utils.commandline import (
    CmdOption,
    ConfigurationGenerator,
    InteractiveField,
    setup_rsmcmd_parser,
)
from rsmtool.utils.constants import CHECK_FIELDS, DEFAULTS, INTERACTIVE_MODE_METADATA
from rsmtool.utils.conversion import convert_to_float, int_to_float, parse_range
from rsmtool.utils.cross_validation import (
    combine_fold_prediction_files,
    create_xval_files,
)
from rsmtool.utils.files import (
    get_output_directory_extension,
    has_files_with_extension,
    parse_json_with_comments,
)
from rsmtool.utils.logging import get_file_logger
from rsmtool.utils.metrics import (
    compute_expected_scores_from_model,
    difference_of_standardized_means,
    partial_correlations,
    quadratic_weighted_kappa,
    standardized_mean_difference,
)
from rsmtool.utils.notebook import (
    bold_highlighter,
    color_highlighter,
    compute_subgroup_plot_params,
    custom_highlighter,
    float_format_func,
    get_files_as_html,
    get_thumbnail_as_html,
    int_or_float_format_func,
)
from rsmtool.writer import DataWriter

# allow test directory to be set via an environment variable
# which is needed for package testing
TEST_DIR = environ.get("TESTDIR", None)
if TEST_DIR:
    rsmtool_test_dir = TEST_DIR
else:
    from rsmtool.test_utils import rsmtool_test_dir


def test_int_to_float():
    eq_(int_to_float(5), 5.0)
    eq_(int_to_float("5"), "5")
    eq_(int_to_float(5.0), 5.0)


def test_convert_to_float():
    eq_(convert_to_float(5), 5.0)
    eq_(convert_to_float("5"), 5.0)
    eq_(convert_to_float(5.0), 5.0)


def test_parse_range():
    eq_(parse_range("5-10"), [5, 10])
    eq_(parse_range("0-100"), [0, 100])


@raises(ValueError)
def test_parse_range_invalid_1():
    parse_range("10")


@raises(ValueError)
def test_parse_range_invalid_2():
    parse_range("10-")


@raises(ValueError)
def test_parse_range_invalid_3():
    parse_range("10-1")


@raises(ValueError)
def test_parse_range_invalid_4():
    parse_range("-10")


def test_parse_json_with_comments():
    json_with_comments = """
    {{
        "key1": "{0}", {1}
        {1}
        "key2": "value2",
        "key3": 5
    }}"""

    # ensure that URLs are not filtered out
    for value in [
        "value1",
        "http://stash.research.ets.org:8000",
        "https://stash.research.ets.org:8000",
    ]:
        for comment in ["/*some comments */", "//some comments "]:
            test_json_string = json_with_comments.format(value, comment)
            expected_result = {"key1": value, "key2": "value2", "key3": 5}
            yield check_parse_json_with_comments, test_json_string, expected_result


def test_parse_json_with_comments_no_comments():
    test_json_string = r"""
    {
        "advisories": {},
        "metadata": {
            "program": null,
            "score_max": 5,
            "score_min": 1
        },
        "multiple_models": false,
        "predict": {
            "trim_max": null,
            "trim_min": null
        },
        "read_data": {
            "data_to_plaintext": [
                "textpipes.pipes.CharFixerDefault",
                "textpipes.pipes.ReplaceTabsWithSpace",
                "textpipes.pipes.MultiSpaceToSingleSpace",
                "textpipes.pipes.Strip"
            ]
        },
        "text_to_instance": null,
        "train": {
            "notes": "The training process was recorded here: https://foobar.org/blah",
            "repository_commit": "12.2020_model_deployment_1.0",
            "repository_name": "strides-scoring-pt",
            "repository_url": "https://stash.research.ets.org:7994/BR/repos/strides/browse",
            "train_cmd": "futil.run_experiment --config foo.jsonnnet"
        }
    }"""

    expected_result = {
        "advisories": {},
        "metadata": {"program": None, "score_max": 5, "score_min": 1},
        "multiple_models": False,
        "predict": {"trim_max": None, "trim_min": None},
        "read_data": {
            "data_to_plaintext": [
                "textpipes.pipes.CharFixerDefault",
                "textpipes.pipes.ReplaceTabsWithSpace",
                "textpipes.pipes.MultiSpaceToSingleSpace",
                "textpipes.pipes.Strip",
            ]
        },
        "text_to_instance": None,
        "train": {
            "notes": r"The training process was recorded here: https://foobar.org/blah",
            "repository_commit": "12.2020_model_deployment_1.0",
            "repository_name": "strides-scoring-pt",
            "repository_url": "https://stash.research.ets.org:7994/BR/repos/strides/browse",
            "train_cmd": "futil.run_experiment --config foo.jsonnnet",
        },
    }
    yield check_parse_json_with_comments, test_json_string, expected_result


def check_parse_json_with_comments(test_json_string, expected_result):
    tempf = NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    filename = tempf.name
    tempf.write(test_json_string)
    tempf.close()

    result = parse_json_with_comments(filename)

    # get rid of the file now that have read it into memory
    unlink(filename)
    eq_(result, expected_result)


def test_float_format_func_default_prec():
    x = 1 / 3
    ans = "0.333"
    assert_equal(float_format_func(x), ans)


def test_float_format_func_custom_prec():
    x = 1 / 3
    ans = "0.3"
    assert_equal(float_format_func(x, 1), ans)


def test_float_format_func_add_extra_zeros():
    x = 0.5
    ans = "0.500"
    assert_equal(float_format_func(x), ans)


def test_float_format_func_add_scientific_not_apply():
    x = 0.5
    ans = "0.500"
    assert_equal(float_format_func(x, scientific=True), ans)


def test_float_format_func_add_scientific_apply():
    x = 0.0003
    ans = "3.000e-04"
    assert_equal(float_format_func(x, scientific=True), ans)


def test_int_or_float_format_func_with_integer_as_float():
    x = 3.0
    ans = "3"
    assert_equal(int_or_float_format_func(x), ans)


def test_int_or_float_format_func_with_float_and_custom_precision():
    x = 1 / 3
    ans = "0.33"
    assert_equal(int_or_float_format_func(x, 2), ans)


def test_custom_highlighter_not_bold_default_values():
    x = 1 / 3
    ans = "0.333"
    assert_equal(custom_highlighter(x), ans)


def test_custom_highlighter_bold_default_values():
    x = -1 / 3
    ans = '<span class="highlight_bold">-0.333</span>'
    assert_equal(custom_highlighter(x), ans)


def test_custom_highlighter_bold_custom_low():
    x = 1 / 3
    ans = '<span class="highlight_bold">0.333</span>'
    assert_equal(custom_highlighter(x, low=0.5), ans)


def test_custom_highlighter_bold_custom_high():
    x = 1 / 3
    ans = '<span class="highlight_bold">0.333</span>'
    assert_equal(custom_highlighter(x, high=0.2), ans)


def test_custom_highlighter_bold_custom_prec():
    x = -1 / 3
    ans = '<span class="highlight_bold">-0.3</span>'
    assert_equal(custom_highlighter(x, prec=1), ans)


def test_custom_highlighter_bold_use_absolute():
    x = -4 / 3
    ans = '<span class="highlight_bold">-1.333</span>'
    assert_equal(custom_highlighter(x, absolute=True), ans)


def test_custom_highlighter_not_bold_custom_low():
    x = -1 / 3
    ans = "-0.333"
    assert_equal(custom_highlighter(x, low=-1), ans)


def test_custom_highlighter_not_bold_custom_high():
    x = 1 / 3
    ans = "0.333"
    assert_equal(custom_highlighter(x, high=0.34), ans)


def test_custom_highlighter_not_bold_custom_prec():
    x = 1 / 3
    ans = "0.3"
    assert_equal(custom_highlighter(x, prec=1), ans)


def test_custom_highlighter_not_bold_use_absolute():
    x = -1 / 3
    ans = "-0.333"
    assert_equal(custom_highlighter(x, absolute=True), ans)


def test_custom_highlighter_not_colored_default_values():
    x = 1 / 3
    ans = "0.333"
    assert_equal(custom_highlighter(x, span_class="color"), ans)


def test_custom_highlighter_color_default_values():
    x = -1 / 3
    ans = '<span class="highlight_color">-0.333</span>'
    assert_equal(custom_highlighter(x, span_class="color"), ans)


def test_bold_highlighter_custom_values_not_bold():
    x = -100.33333
    ans = "-100.3"
    assert_equal(bold_highlighter(x, 100, 101, 1, absolute=True), ans)


def test_bold_highlighter_custom_values_bold():
    x = -100.33333
    ans = '<span class="highlight_bold">-100.3</span>'
    assert_equal(bold_highlighter(x, 99, 100, 1, absolute=True), ans)


def test_color_highlighter_custom_values_not_color():
    x = -100.33333
    ans = "-100.3"
    assert_equal(color_highlighter(x, 100, 101, 1, absolute=True), ans)


def test_color_highlighter_custom_values_color():
    x = -100.33333
    ans = '<span class="highlight_color">-100.3</span>'
    assert_equal(color_highlighter(x, 99, 100, 1, absolute=True), ans)


def test_compute_subgroup_params_with_two_groups():
    figure_width = 4
    figure_height = 8
    num_rows, num_cols = 2, 2
    group_names = ["A", "B"]

    expected_subgroup_plot_params = (
        figure_width,
        figure_height,
        num_rows,
        num_cols,
        group_names,
    )

    subgroup_plot_params = compute_subgroup_plot_params(group_names, 3)
    eq_(expected_subgroup_plot_params, subgroup_plot_params)


def test_compute_subgroup_params_with_10_groups():
    figure_width = 10
    figure_height = 18
    num_rows, num_cols = 3, 1
    group_names = [i for i in range(10)]
    wrapped_group_names = [str(i) for i in group_names]

    expected_subgroup_plot_params = (
        figure_width,
        figure_height,
        num_rows,
        num_cols,
        wrapped_group_names,
    )

    subgroup_plot_params = compute_subgroup_plot_params(group_names, 3)
    eq_(expected_subgroup_plot_params, subgroup_plot_params)


def test_compute_subgroups_with_wrapping_and_five_plots():
    figure_width = 10
    figure_height = 30
    num_rows, num_cols = 5, 1
    group_names = [
        f"this is a very long string that will ultimately be wrapped I assume {i}"
        for i in range(10)
    ]

    wrapped_group_names = [
        f"this is a very long\nstring that will\nultimately be\nwrapped I assume {i}"
        for i in range(10)
    ]

    expected_subgroup_plot_params = (
        figure_width,
        figure_height,
        num_rows,
        num_cols,
        wrapped_group_names,
    )

    subgroup_plot_params = compute_subgroup_plot_params(group_names, 5)
    eq_(expected_subgroup_plot_params, subgroup_plot_params)


def test_has_files_with_extension_true():
    directory = join(rsmtool_test_dir, "data", "files")
    result = has_files_with_extension(directory, "csv")
    eq_(result, True)


def test_has_files_with_extension_false():
    directory = join(rsmtool_test_dir, "data", "files")
    result = has_files_with_extension(directory, "ppt")
    eq_(result, False)


def test_get_output_directory_extension():
    directory = join(rsmtool_test_dir, "data", "experiments", "lr", "output")
    result = get_output_directory_extension(directory, "id_1")
    eq_(result, "csv")


@raises(ValueError)
def test_get_output_directory_extension_error():
    directory = join(rsmtool_test_dir, "data", "files")
    get_output_directory_extension(directory, "id_1")


def test_standardized_mean_difference():
    # test SMD
    expected = 1 / 4
    smd = standardized_mean_difference(8, 9, 4, 4, method="williamson")
    eq_(smd, expected)


def test_standardized_mean_difference_zero_denominator_johnson():
    # test SMD with zero denominator
    # we pass 0 as standard deviation of population
    # and use Johnson method
    # which uses it as denominator
    smd = standardized_mean_difference([3.2, 3.5], [4.2, 3.1], 0, 0, method="Johnson")
    assert np.isnan(smd)


def test_standardized_mean_difference_zero_difference():
    # test SMD with zero difference between groups
    expected = 0.0
    smd = standardized_mean_difference(4.2, 4.2, 1.1, 1.1, method="williamson")
    eq_(smd, expected)


@raises(ValueError)
def test_standardized_mean_difference_fake_method():
    # test SMD with fake method
    standardized_mean_difference(4.2, 4.2, 1.1, 1.1, method="foobar")


def test_standardized_mean_difference_pooled():
    expected = 0.8523247028586811
    smd = standardized_mean_difference([8, 4, 6, 3], [9, 4, 5, 12], method="pooled", ddof=0)
    eq_(smd, expected)


def test_standardized_mean_difference_unpooled():
    expected = 1.171700198827415
    smd = standardized_mean_difference([8, 4, 6, 3], [9, 4, 5, 12], method="unpooled", ddof=0)
    eq_(smd, expected)


def test_standardized_mean_difference_johnson():
    expected = 0.9782608695652175
    smd = standardized_mean_difference(
        [8, 4, 6, 3],
        [9, 4, 5, 12],
        method="johnson",
        population_y_true_observed_sd=2.3,
        ddof=0,
    )
    eq_(smd, expected)


@raises(ValueError)
def test_standardized_mean_difference_johnson_error():
    standardized_mean_difference([8, 4, 6, 3], [9, 4, 5, 12], method="johnson", ddof=0)


@raises(AssertionError)
def test_difference_of_standardized_means_unequal_lengths():
    difference_of_standardized_means([8, 4, 6, 3], [9, 4, 5, 12, 17])


@raises(ValueError)
def test_difference_of_standardized_means_with_y_true_mn_but_no_sd():
    difference_of_standardized_means([8, 4, 6, 3], [9, 4, 5, 12], population_y_true_observed_mn=4.5)


@raises(ValueError)
def test_difference_of_standardized_means_with_y_true_sd_but_no_mn():
    difference_of_standardized_means([8, 4, 6, 3], [9, 4, 5, 12], population_y_true_observed_sd=1.5)


@raises(ValueError)
def test_difference_of_standardized_means_with_y_pred_mn_but_no_sd():
    difference_of_standardized_means([8, 4, 6, 3], [9, 4, 5, 12], population_y_pred_mn=4.5)


@raises(ValueError)
def test_difference_of_standardized_means_with_y_pred_sd_but_no_mn():
    difference_of_standardized_means([8, 4, 6, 3], [9, 4, 5, 12], population_y_pred_sd=1.5)


def test_difference_of_standardized_means_with_all_values():
    expected = 0.7083333333333336
    y_true, y_pred = np.array([8, 4, 6, 3]), np.array([9, 4, 5, 12])
    diff_std_means = difference_of_standardized_means(
        y_true,
        y_pred,
        population_y_true_observed_mn=4.5,
        population_y_pred_mn=5.1,
        population_y_true_observed_sd=1.2,
        population_y_pred_sd=1.8,
    )
    eq_(diff_std_means, expected)


def test_difference_of_standardized_means_with_no_population_info():
    # this test is expected to raise two UserWarning
    # because we did not pass population means for y_true and y_pred
    expected = -1.7446361815538174e-16
    y_true, y_pred = (
        np.array([98, 18, 47, 64, 32, 11, 100]),
        np.array([94, 42, 54, 12, 92, 10, 77]),
    )
    with warnings.catch_warnings(record=True) as warning_list:
        diff_std_means = difference_of_standardized_means(y_true, y_pred)
    eq_(diff_std_means, expected)
    eq_(len(warning_list), 2)
    assert issubclass(warning_list[0].category, UserWarning)
    assert issubclass(warning_list[1].category, UserWarning)


def test_difference_of_standardized_means_zero_population_sd_pred():
    y_true, y_pred = (
        np.array([3, 5, 1, 2, 2, 3, 1, 4, 1, 2]),
        np.array([2, 1, 4, 1, 5, 2, 2, 2, 2, 2]),
    )
    expected = None
    diff_std_means = difference_of_standardized_means(
        y_true,
        y_pred,
        population_y_true_observed_mn=2.44,
        population_y_true_observed_sd=0.54,
        population_y_pred_mn=2.44,
        population_y_pred_sd=0,
    )
    eq_(diff_std_means, expected)


def test_difference_of_standardized_means_zero_population_sd_human():
    y_true, y_pred = (
        np.array([3, 5, 1, 2, 2, 3, 1, 4, 1, 2]),
        np.array([2, 1, 4, 1, 5, 2, 2, 2, 2, 2]),
    )
    expected = None
    diff_std_means = difference_of_standardized_means(
        y_true,
        y_pred,
        population_y_pred_mn=2.44,
        population_y_pred_sd=0.54,
        population_y_true_observed_mn=2.44,
        population_y_true_observed_sd=0,
    )
    eq_(diff_std_means, expected)


def test_difference_of_standardized_means_zero_population_computed():
    # sd is computed from the data and is zero
    y_pred, y_true = (
        np.array([3, 5, 1, 2, 2, 3, 1, 4, 1, 2]),
        np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
    )
    expected = None
    diff_std_means = difference_of_standardized_means(y_true, y_pred)
    eq_(diff_std_means, expected)


def test_quadratic_weighted_kappa():
    expected_qwk = -0.09210526315789469
    computed_qwk = quadratic_weighted_kappa(np.array([8, 4, 6, 3]), np.array([9, 4, 5, 12]))
    assert_almost_equal(computed_qwk, expected_qwk)


def test_quadratic_weighted_kappa_discrete_values_match_skll():
    data = (np.array([8, 4, 6, 3]), np.array([9, 4, 5, 12]))
    qwk_rsmtool = quadratic_weighted_kappa(data[0], data[1])
    qwk_skll = kappa(data[0], data[1], weights="quadratic")
    assert_almost_equal(qwk_rsmtool, qwk_skll)


def test_quadratic_weighted_kappa_discrete_values_match_sklearn():
    data = (np.array([8, 4, 6, 3]), np.array([9, 4, 5, 12]))
    qwk_rsmtool = quadratic_weighted_kappa(data[0], data[1])
    qwk_sklearn = cohen_kappa_score(
        data[0], data[1], weights="quadratic", labels=[3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    )
    assert_almost_equal(qwk_rsmtool, qwk_sklearn)


@raises(AssertionError)
def test_quadratic_weighted_kappa_error():
    quadratic_weighted_kappa(np.array([8, 4, 6, 3]), np.array([9, 4, 5, 12, 11]))


def test_partial_correlations_with_singular_matrix():
    # This test is expected to pass UserWarning becaus
    # of singularity
    expected = pd.DataFrame({0: [1.0, -1.0], 1: [-1.0, 1.0]})
    df_singular = pd.DataFrame(np.tile(np.random.randn(100), (2, 1))).T
    with warnings.catch_warnings(record=True) as warning_list:
        assert_frame_equal(partial_correlations(df_singular), expected)
    eq_(len(warning_list), 1)
    assert issubclass(warning_list[-1].category, UserWarning)


def test_partial_correlations_pinv():
    msg = (
        "When computing partial correlations "
        "the inverse of the variance-covariance matrix "
        "was calculated "
        "using the Moore-Penrose generalized matrix inversion, due to "
        "its determinant being at or very close to zero."
    )
    df_small_det = pd.DataFrame(
        {
            "X1": [1.3, 1.2, 1.5, 1.7, 1.8, 1.9, 2.0],
            "X2": [1.3, 1.2, 1.5, 1.7001, 1.8, 1.9, 2.0],
        }
    )

    with warnings.catch_warnings(record=True) as wrn:
        warnings.simplefilter("always")
        partial_correlations(df_small_det)
        eq_(str(wrn[-1].message), msg)


class TestLogging:
    def setUp(self):
        # create a log file with a random name
        logname = str(uuid4())
        self.logpath = Path(rsmtool_test_dir) / f"{logname}.log"

    def tearDown(self):
        try:
            unlink(self.logpath)
        except PermissionError:
            pass  # sometimes Azure has trouble deleting this file

    def test_get_file_logger(self):
        # create a logger based on our dummy log file
        test_logger = get_file_logger("testing", self.logpath)

        # log message to that file
        test_logger.info("This is a test.")

        # delete the logger
        del test_logger

        # check that the message was indeed written
        with open(self.logpath, "r") as tempfh:
            ok_(tempfh.read().strip(), "This is a test.")


class TestCrossValidation:
    def check_create_xval_files(
        self, file_format, with_folds_file, with_feature_subset, with_feature_list
    ):
        """
        Check that ``create_xval_files()`` functions as expected.

        Parameters
        ----------
        file_format : str
            File format to use for testing. One of {"csv", "tsv", "xlsx"}.
        with_folds_file : bool
            Whether to use a pre-specified folds file for testing.
        with_feature_subset : bool
            Whether to use feature subset information for testing.
        with_feature_list : bool
            Whether to specify features as a list instead of a file
        """
        # create an rsmxval configuration dictionary
        rsmxval_config_dict = {
            "train_file": join(rsmtool_test_dir, "data", "files", f"train.{file_format}"),
            "file_format": file_format,
            "id_column": "ID",
            "model": "LinearRegression",
            "train_label_column": "score",
            "experiment_id": "test_create_xval_files",
            "description": "Test Create Xval Files",
        }

        # use a folds file if asked, otherwise a fixed number of folds
        if with_folds_file:
            rsmxval_config_dict["folds_file"] = join(rsmtool_test_dir, "data", "files", "folds.csv")
        else:
            rsmxval_config_dict["folds"] = 7

        # use a feature subset if asked
        if with_feature_subset:
            rsmxval_config_dict["feature_subset_file"] = join(
                rsmtool_test_dir,
                "data",
                "experiments",
                "lr-with-feature-subset-file-and-feature-file",
                "feature_file.csv",
            )
            rsmxval_config_dict["feature_subset"] = "subset1"
        # otherwise set "features" depending on whether it should be a list or not
        else:
            if with_feature_list:
                rsmxval_config_dict["features"] = ["FEATURE1", "FEATURE2"]
            else:
                rsmxval_config_dict["features"] = join(
                    rsmtool_test_dir, "data", "experiments", "lr", "features.csv"
                )

        # create configuration object from dictionary
        rsmxval_config = Configuration(rsmxval_config_dict, context="rsmxval")

        # read the original training data file
        df_train_expected = DataReader.read_from_file(rsmxval_config.get("train_file"))

        # create a temporary output directory and any sub-directories
        # that are needed by the ``create_xval_files()`` function
        output_dir = mkdtemp()
        foldsdir = Path(output_dir) / "folds"
        modeldir = Path(output_dir) / "final-model"
        makedirs(foldsdir)
        makedirs(modeldir)

        # call the function
        df_train_actual, expected_folds = create_xval_files(rsmxval_config, output_dir)

        # check that the training data frame is as expected
        assert_frame_equal(df_train_actual, df_train_expected)

        # check that there are only the expected number of fold subdirectories
        actual_foldsdir_contents = sorted(listdir(foldsdir))
        expected_foldsdir_contents = [f"{fold_num:02}" for fold_num in range(1, expected_folds + 1)]
        eq_(actual_foldsdir_contents, expected_foldsdir_contents)

        # check all the per-fold files/directories
        for fold_subdir in foldsdir.iterdir():
            # (a) per-fold subdirectories and configuration files
            fold_num = fold_subdir.name
            fold_config = fold_subdir / "rsmtool.json"
            ok_(fold_subdir.exists() and fold_subdir.is_dir())
            ok_(fold_config.exists() and fold_config.is_file())

            # (b) per-fold train/test files exist and are subsets of
            #     the original full training file
            fold_train_file = fold_subdir / f"train.{file_format}"
            fold_test_file = fold_subdir / f"test.{file_format}"
            ok_(fold_train_file.exists() and fold_train_file.is_file())
            ok_(fold_test_file.exists() and fold_test_file.is_file())
            df_train_actual_fold = DataReader.read_from_file(fold_train_file)
            df_test_actual_fold = DataReader.read_from_file(fold_test_file)
            eq_(
                len(df_train_expected),
                len(df_train_actual_fold) + len(df_test_actual_fold),
            )
            id_column = rsmxval_config.get("id_column")
            assert_array_equal(
                df_train_expected[id_column].values.sort(),
                np.concatenate(
                    [
                        df_train_actual_fold[id_column].values,
                        df_test_actual_fold[id_column].values,
                    ]
                ).sort(),
            )

            # (c) configuration file fields
            parsed_fold_config = ConfigurationParser(str(fold_config)).parse(context="rsmtool")

            eq_(Path(parsed_fold_config.get("train_file")), fold_train_file)

            eq_(Path(parsed_fold_config.get("test_file")), fold_test_file)

            eq_(parsed_fold_config.get("id_column"), rsmxval_config.get("id_column"))

            eq_(parsed_fold_config.get("id_column"), rsmxval_config.get("id_column"))

            eq_(
                parsed_fold_config.get("train_label_column"),
                rsmxval_config.get("train_label_column"),
            )

            eq_(
                parsed_fold_config.get("test_label_column"),
                rsmxval_config.get("train_label_column"),
            )

            eq_(parsed_fold_config.get("file_format"), rsmxval_config.get("file_format"))

            eq_(parsed_fold_config.get("features"), rsmxval_config.get("features"))

            eq_(
                parsed_fold_config.get("experiment_id"),
                f"{rsmxval_config.get('experiment_id')}_fold{fold_num}",
            )

            eq_(
                parsed_fold_config.get("description"),
                f"{rsmxval_config.get('description')} (Fold {fold_num})",
            )

            # (d) the per-fold features or feature subset files
            if with_feature_subset:
                subset_file = fold_subdir / Path(rsmxval_config.get("feature_subset_file")).name
                subset_name = rsmxval_config.get("feature_subset")
                ok_(subset_file.exists() and subset_file.is_file())
                eq_(parsed_fold_config.get("feature_subset"), subset_name)
                ok_(filecmp.cmp(subset_file, rsmxval_config.get("feature_subset_file")))
            else:
                # if "features" a list, it should be in the config
                if with_feature_list:
                    eq_(parsed_fold_config["features"], ["FEATURE1", "FEATURE2"])
                # otherwise the "features" file should have been copied to the fold directory
                else:
                    fold_feature_file = fold_subdir / Path(rsmxval_config.get("features")).name
                    ok_(fold_feature_file.exists() and fold_feature_file.is_file())
                    ok_(filecmp.cmp(fold_feature_file, rsmxval_config.get("features")))

        # (e) the dummy test file for the final model
        dummy_test_file = modeldir / f"dummy_test.{file_format}"
        ok_(dummy_test_file.exists() and dummy_test_file.is_file())

        # remove the entire output directory tree
        rmtree(output_dir)

    def test_create_xval_files(self):
        for (
            file_format,
            with_folds_file,
            with_feature_subset,
            with_feature_list,
        ) in product(["csv", "tsv", "xlsx"], [False, True], [False, True], [False, True]):
            yield (
                self.check_create_xval_files,
                file_format,
                with_folds_file,
                with_feature_subset,
                with_feature_list,
            )

    def check_combine_fold_prediction_files(self, file_format):
        """
        Check that ``combine_fold_prediction_files()`` functions as expected.

        Parameters
        ----------
        file_format : str
            File format to use for testing. One of {"csv", "tsv", "xlsx"}.
        """
        # create a temporary output directory and any sub-directories
        # that are needed by the ``combine_fold_prediction_files()`` function
        output_dir = mkdtemp()
        foldsdir = Path(output_dir) / "folds"
        makedirs(foldsdir)

        # create 3 sub-directories simulating 3-fold cross-validation
        makedirs(foldsdir / "01")
        makedirs(foldsdir / "02")
        makedirs(foldsdir / "03")

        # create prediction files in each of the fold sub-directories
        df_preds_fold1 = pd.DataFrame(np.random.normal(size=(30, 2)), columns=["raw", "scale"])
        df_preds_fold1["spkitemid"] = [f"RESPONSE_{i}" for i in range(1, 31)]
        df_preds_fold2 = pd.DataFrame(np.random.normal(size=(30, 2)), columns=["raw", "scale"])
        df_preds_fold2["spkitemid"] = [f"RESPONSE_{i}" for i in range(31, 61)]
        df_preds_fold3 = pd.DataFrame(np.random.normal(size=(30, 2)), columns=["raw", "scale"])
        df_preds_fold3["spkitemid"] = [f"RESPONSE_{i}" for i in range(61, 91)]

        # combine each of the frames in memory
        df_combined_expected = pd.concat(
            [df_preds_fold1, df_preds_fold2, df_preds_fold3], keys="spkitemid"
        )

        DataWriter.write_frame_to_file(
            df_preds_fold1,
            str(foldsdir / "01" / "pred_processed"),
            file_format=file_format,
        )
        DataWriter.write_frame_to_file(
            df_preds_fold2,
            str(foldsdir / "02" / "pred_processed"),
            file_format=file_format,
        )
        DataWriter.write_frame_to_file(
            df_preds_fold3,
            str(foldsdir / "03" / "pred_processed"),
            file_format=file_format,
        )

        # now call `combine_fold_prediction_files` and check that its output
        # matches the frame that we manually combined
        df_combined_actual = combine_fold_prediction_files(str(foldsdir), file_format)
        assert_frame_equal(
            df_combined_expected.sort_values(by="spkitemid").reset_index(drop=True),
            df_combined_actual.sort_values(by="spkitemid").reset_index(drop=True),
        )

        # delete the temporary directory and all sub-directories
        rmtree(output_dir)

    def test_combine_fold_prediction_files(self):
        yield self.check_combine_fold_prediction_files, "csv"
        yield self.check_combine_fold_prediction_files, "tsv"
        yield self.check_combine_fold_prediction_files, "xlsx"


class TestIntermediateFiles:
    def get_files(self, file_format="csv"):
        directory = join(rsmtool_test_dir, "data", "output")
        files = sorted([f for f in listdir(directory) if f.endswith(file_format)])
        return files, directory

    def test_get_files_as_html(self):
        files, directory = self.get_files()
        html_string = (
            """<li><b>Betas</b>: <a href="{}" download>csv</a></li>"""
            """<li><b>Eval</b>: <a href="{}" download>csv</a></li>"""
        )

        html_expected = html_string.format(
            join("..", "output", files[0]), join("..", "output", files[1])
        )
        html_expected = "".join(html_expected.strip().split())
        html_expected = """<ul><html>""" + html_expected + """</ul></html>"""
        html_result = get_files_as_html(directory, "lr", "csv")
        html_result = "".join(html_result.strip().split())
        eq_(html_expected, html_result)

    def test_get_files_as_html_replace_dict(self):
        files, directory = self.get_files()
        html_string = (
            """<li><b>THESE BETAS</b>: <a href="{}" download>csv</a></li>"""
            """<li><b>THESE EVALS</b>: <a href="{}" download>csv</a></li>"""
        )

        replace_dict = {"betas": "THESE BETAS", "eval": "THESE EVALS"}
        html_expected = html_string.format(
            join("..", "output", files[0]), join("..", "output", files[1])
        )
        html_expected = "".join(html_expected.strip().split())
        html_expected = """<ul><html>""" + html_expected + """</ul></html>"""
        html_result = get_files_as_html(directory, "lr", "csv", replace_dict)
        html_result = "".join(item for item in html_result)
        html_result = "".join(html_result.strip().split())
        eq_(html_expected, html_result)


class TestThumbnail:
    def get_result(self, path, id_num="1", other_path=None):
        if other_path is None:
            other_path = path

        # get the expected HTML output

        result = f"""
        <img id='{id_num}' src='{path}'
        onclick='getPicture("{other_path}")'
        title="Click to enlarge">
        </img>
        <style>
        img {{
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
            width: 150px;
            cursor: pointer;
        }}
        </style>

        <script>
        function getPicture(picpath) {{
            window.open(picpath, 'Image', resizable=1);
        }};
        </script>"""
        return "".join(result.strip().split())

    def test_convert_to_html(self):
        # simple test of HTML thumbnail conversion

        path = relpath(join(rsmtool_test_dir, "data", "figures", "figure1.svg"))
        image = get_thumbnail_as_html(path, 1)

        clean_image = "".join(image.strip().split())
        clean_thumb = self.get_result(path)

        eq_(clean_image, clean_thumb)

    def test_convert_to_html_with_png(self):
        # simple test of HTML thumbnail conversion
        # with a PNG file instead of SVG

        path = relpath(join(rsmtool_test_dir, "data", "figures", "figure3.png"))
        image = get_thumbnail_as_html(path, 1)

        clean_image = "".join(image.strip().split())
        clean_thumb = self.get_result(path)

        eq_(clean_image, clean_thumb)

    def test_convert_to_html_with_two_images(self):
        # test converting two images to HTML thumbnails

        path1 = relpath(join(rsmtool_test_dir, "data", "figures", "figure1.svg"))
        path2 = relpath(join(rsmtool_test_dir, "data", "figures", "figure2.svg"))

        counter = count(1)
        image = get_thumbnail_as_html(path1, next(counter))
        image = get_thumbnail_as_html(path2, next(counter))

        clean_image = "".join(image.strip().split())
        clean_thumb = self.get_result(path2, 2)

        eq_(clean_image, clean_thumb)

    def test_convert_to_html_with_absolute_path(self):
        # test converting image to HTML with absolute path

        path = relpath(join(rsmtool_test_dir, "data", "figures", "figure1.svg"))
        path_absolute = abspath(path)

        image = get_thumbnail_as_html(path_absolute, 1)

        clean_image = "".join(image.strip().split())
        clean_thumb = self.get_result(path)

        eq_(clean_image, clean_thumb)

    @raises(FileNotFoundError)
    def test_convert_to_html_file_not_found_error(self):
        # test FileNotFound error properly raised

        path = "random/path/asftesfa/to/figure1.svg"
        get_thumbnail_as_html(path, 1)

    def test_convert_to_html_with_different_thumbnail(self):
        # test converting image to HTML with different thumbnail

        path1 = relpath(join(rsmtool_test_dir, "data", "figures", "figure1.svg"))
        path2 = relpath(join(rsmtool_test_dir, "data", "figures", "figure2.svg"))

        image = get_thumbnail_as_html(path1, 1, path_to_thumbnail=path2)

        clean_image = "".join(image.strip().split())
        clean_thumb = self.get_result(path1, other_path=path2)

        eq_(clean_image, clean_thumb)

    @raises(FileNotFoundError)
    def test_convert_to_html_thumbnail_not_found_error(self):
        # test FileNotFound error properly raised for thumbnail

        path1 = relpath(join(rsmtool_test_dir, "data", "figures", "figure1.svg"))
        path2 = "random/path/asftesfa/to/figure1.svg"
        _ = get_thumbnail_as_html(path1, 1, path_to_thumbnail=path2)


class TestExpectedScores:
    @classmethod
    def setUpClass(cls):
        # create a dummy train and test feature set
        X, y = make_classification(
            n_samples=525, n_features=10, n_classes=5, n_informative=8, random_state=123
        )
        X_train, y_train = X[:500], y[:500]
        X_test = X[500:]

        train_ids = list(range(1, len(X_train) + 1))
        train_features = [
            dict(zip([f"FEATURE_{i + 1}" for i in range(X_train.shape[1])], x)) for x in X_train
        ]
        train_labels = list(y_train)

        test_ids = list(range(1, len(X_test) + 1))
        test_features = [
            dict(zip([f"FEATURE_{i + 1}" for i in range(X_test.shape[1])], x)) for x in X_test
        ]

        cls.train_fs = FeatureSet(
            "train", ids=train_ids, features=train_features, labels=train_labels
        )
        cls.test_fs = FeatureSet("test", ids=test_ids, features=test_features)

        # train some test SKLL learners that we will use in our tests

        # we catch convergence warnings since the model doesn't converge
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            cls.linearsvc = Learner("LinearSVC")
            _ = cls.linearsvc.train(cls.train_fs, grid_search=False)

        cls.svc = Learner("SVC")
        _ = cls.svc.train(cls.train_fs, grid_search=False)

        cls.svc_with_probs = Learner("SVC", probability=True)
        _ = cls.svc_with_probs.train(cls.train_fs, grid_search=False)

    @raises(ValueError)
    def test_wrong_model(self):
        compute_expected_scores_from_model(self.linearsvc, self.test_fs, 0, 4)

    @raises(ValueError)
    def test_svc_model_trained_with_no_probs(self):
        compute_expected_scores_from_model(self.svc, self.test_fs, 0, 4)

    @raises(ValueError)
    def test_wrong_score_range(self):
        compute_expected_scores_from_model(self.svc_with_probs, self.test_fs, 0, 3)

    def test_expected_scores(self):
        computed_predictions = compute_expected_scores_from_model(
            self.svc_with_probs, self.test_fs, 0, 4
        )
        assert len(computed_predictions) == len(self.test_fs)
        assert np.all(
            [((prediction >= 0) and (prediction <= 4)) for prediction in computed_predictions]
        )


class TestCmdOption:
    @raises(TypeError)
    def test_cmd_option_no_help(self):
        """Test that CmdOption with no help raises exception."""
        _ = CmdOption(longname="foo", dest="blah")

    @raises(TypeError)
    def test_cmd_option_no_dest(self):
        """Test that CmdOption with no dest raises exception."""
        _ = CmdOption(longname="foo", help="this option has no dest")

    def test_cmd_option_attributes(self):
        """Test CmdOption attributes."""
        co = CmdOption(dest="good", help="this option has only dest and help")
        eq_(co.dest, "good")
        eq_(co.help, "this option has only dest and help")
        ok_(co.action is None)
        ok_(co.longname is None)
        ok_(co.shortname is None)
        ok_(co.required is None)
        ok_(co.nargs is None)
        ok_(co.default is None)


class TestSetupRsmCmdParser:
    def test_run_subparser_no_args(self):
        """Test run subparser with no arguments."""
        parser = setup_rsmcmd_parser("test")
        # we need to patch sys.exit since --help just exists otherwise
        with patch("sys.exit") as exit_mock:
            parsed_namespace = parser.parse_args("run --help".split())
        expected_namespace = argparse.Namespace(
            config_file=None, output_dir=getcwd(), subcommand="run"
        )
        eq_(parsed_namespace, expected_namespace)
        assert exit_mock.called

    @raises(SystemExit)
    def test_run_subparser_non_existent_config_file(self):
        """Test run subparser with a non-existent config file."""
        parser = setup_rsmcmd_parser("test")
        _ = parser.parse_args("run fake.json".split())

    def test_run_subparser_with_output_directory(self):
        """Test run subparser with a specified output directory."""
        parser = setup_rsmcmd_parser("test")
        config_file = join(rsmtool_test_dir, "data", "experiments", "lr", "lr.json")
        parsed_namespace = parser.parse_args(f"run {config_file} /path/to/output/dir".split())

        expected_namespace = argparse.Namespace(
            config_file=config_file, output_dir="/path/to/output/dir", subcommand="run"
        )
        eq_(parsed_namespace, expected_namespace)

    def test_run_subparser_no_output_directory(self):
        """Test run subparser where no output directory is required."""
        parser = setup_rsmcmd_parser("test", uses_output_directory=False)
        config_file = join(rsmtool_test_dir, "data", "experiments", "lr", "lr.json")
        parsed_namespace = parser.parse_args(f"run {config_file}".split())
        expected_namespace = argparse.Namespace(config_file=config_file, subcommand="run")
        ok_(not hasattr(parsed_namespace, "output_dir"))
        eq_(parsed_namespace, expected_namespace)

    def test_run_subparser_with_overwrite_enabled(self):
        """Test run subparser with overwriting enabled."""
        parser = setup_rsmcmd_parser("test", allows_overwriting=True)
        config_file = join(rsmtool_test_dir, "data", "experiments", "lr", "lr.json")
        parsed_namespace = parser.parse_args(f"run {config_file} /path/to/output/dir -f".split())
        expected_namespace = argparse.Namespace(
            config_file=config_file,
            output_dir="/path/to/output/dir",
            force_write=True,
            subcommand="run",
        )
        eq_(parsed_namespace, expected_namespace)

    def test_run_subparser_with_extra_options(self):
        """Test run subparser with extra options."""
        extra_options = [
            CmdOption(dest="test_arg", help="a test positional argument"),
            CmdOption(
                shortname="t",
                longname="test",
                dest="test_kwarg",
                help="a test optional argument",
            ),
            CmdOption(
                shortname="x",
                dest="extra_kwarg",
                action="store_true",
                default=False,
                help="a boolean optional argument",
            ),
            CmdOption(
                longname="zeta",
                dest="extra_kwargs2",
                nargs="+",
                required=False,
                help="a multiply specified optional argument",
            ),
        ]
        parser = setup_rsmcmd_parser(
            "test", allows_overwriting=True, extra_run_options=extra_options
        )
        config_file = join(rsmtool_test_dir, "data", "experiments", "lr", "lr.json")
        parsed_namespace = parser.parse_args(
            f"run {config_file} /path/to/output/dir foo --test bar -x --zeta 1 2".split()
        )
        expected_namespace = argparse.Namespace(
            config_file=config_file,
            extra_kwarg=True,
            extra_kwargs2=["1", "2"],
            force_write=False,
            output_dir="/path/to/output/dir",
            subcommand="run",
            test_arg="foo",
            test_kwarg="bar",
        )
        eq_(parsed_namespace, expected_namespace)

    def test_run_subparser_with_extra_options_required_true_not_specified(self):
        """Test run subparser with an unspecified required optional."""
        extra_options = [
            CmdOption(dest="test_arg", help="a test positional argument"),
            CmdOption(
                longname="zeta",
                dest="test_kwargs",
                nargs="+",
                required=True,
                help="a multiply specified optional argument",
            ),
        ]
        parser = setup_rsmcmd_parser(
            "test", uses_output_directory=False, extra_run_options=extra_options
        )
        config_file = join(rsmtool_test_dir, "data", "experiments", "lr", "lr.json")
        with patch("sys.exit") as exit_mock:
            parsed_namespace = parser.parse_args(f"run {config_file} foo".split())
        expected_namespace = argparse.Namespace(
            config_file=config_file, subcommand="run", test_arg="foo", test_kwargs=None
        )
        eq_(parsed_namespace, expected_namespace)
        assert exit_mock.called

    def test_run_subparser_with_extra_options_required_true_and_specified(self):
        """Test run subparser with a specified required optional."""
        extra_options = [
            CmdOption(dest="test_arg", help="a test positional argument"),
            CmdOption(
                longname="zeta",
                dest="test_kwargs",
                nargs="+",
                required=True,
                help="a multiply specified optional argument",
            ),
        ]
        parser = setup_rsmcmd_parser(
            "test", uses_output_directory=False, extra_run_options=extra_options
        )
        config_file = join(rsmtool_test_dir, "data", "experiments", "lr", "lr.json")
        parsed_namespace = parser.parse_args(f"run {config_file} foo --zeta 1 2".split())
        expected_namespace = argparse.Namespace(
            config_file=config_file,
            subcommand="run",
            test_arg="foo",
            test_kwargs=["1", "2"],
        )
        eq_(parsed_namespace, expected_namespace)

    @raises(TypeError)
    def test_run_subparser_with_extra_options_bad_required_value(self):
        """Test run subparser with a non-boolean value for required."""
        extra_options = [
            CmdOption(dest="test_arg", help="a test positional argument"),
            CmdOption(
                longname="zeta",
                dest="test_kwargs",
                nargs="+",
                required="true",
                help="a multiply specified optional argument",
            ),
        ]
        _ = setup_rsmcmd_parser(
            "test", uses_output_directory=False, extra_run_options=extra_options
        )

    def test_generate_subparser_help_flag(self):
        """Test generate subparser with --help specified."""
        parser = setup_rsmcmd_parser("test")
        # we need to patch sys.exit since --help just exists otherwise
        with patch("sys.exit") as exit_mock:
            parsed_namespace = parser.parse_args("generate --help".split())
        expected_namespace = argparse.Namespace(
            subcommand="generate", interactive=False, quiet=False
        )
        eq_(parsed_namespace, expected_namespace)
        assert exit_mock.called

    def test_generate_subparser(self):
        """Test generate subparser with no arguments."""
        parser = setup_rsmcmd_parser("test")
        parsed_namespace = parser.parse_args("generate".split())
        expected_namespace = argparse.Namespace(
            subcommand="generate", interactive=False, quiet=False
        )
        eq_(parsed_namespace, expected_namespace)

    def test_generate_subparser_with_subgroups_and_flag(self):
        """Test generate subparser with subgroups option and flag."""
        parser = setup_rsmcmd_parser("test", uses_subgroups=True)
        parsed_namespace = parser.parse_args("generate --subgroups".split())
        expected_namespace = argparse.Namespace(
            subcommand="generate", interactive=False, quiet=False, subgroups=True
        )
        eq_(parsed_namespace, expected_namespace)

    def test_generate_subparser_with_subgroups_option_and_short_flag(self):
        """Test generate subparser with subgroups option and short flag."""
        parser = setup_rsmcmd_parser("test", uses_subgroups=True)
        parsed_namespace = parser.parse_args("generate -g".split())
        expected_namespace = argparse.Namespace(
            subcommand="generate", interactive=False, quiet=False, subgroups=True
        )
        eq_(parsed_namespace, expected_namespace)

    def test_generate_subparser_with_subgroups_option_but_no_flag(self):
        """Test generate subparser with subgroups option but no flag."""
        parser = setup_rsmcmd_parser("test", uses_subgroups=True)
        parsed_namespace = parser.parse_args("generate".split())
        expected_namespace = argparse.Namespace(
            subcommand="generate", interactive=False, quiet=False, subgroups=False
        )
        eq_(parsed_namespace, expected_namespace)

    def test_generate_subparser_with_only_quiet_flag(self):
        """Test generate subparser with only the quiet flag."""
        parser = setup_rsmcmd_parser("test")
        parsed_namespace = parser.parse_args("generate --quiet".split())
        expected_namespace = argparse.Namespace(
            subcommand="generate", interactive=False, quiet=True
        )
        eq_(parsed_namespace, expected_namespace)

    def test_generate_subparser_with_subgroups_and_quiet_flags(self):
        """Test generate subparser with subgroups and quiet flags."""
        parser = setup_rsmcmd_parser("test", uses_subgroups=True)
        parsed_namespace = parser.parse_args("generate --subgroups -q".split())
        expected_namespace = argparse.Namespace(
            subcommand="generate", interactive=False, quiet=True, subgroups=True
        )
        eq_(parsed_namespace, expected_namespace)

    def test_generate_subparser_with_only_interactive_flag(self):
        """Test generate subparser with only the interactive flag."""
        parser = setup_rsmcmd_parser("test")
        parsed_namespace = parser.parse_args("generate --interactive".split())
        expected_namespace = argparse.Namespace(
            subcommand="generate", interactive=True, quiet=False
        )
        eq_(parsed_namespace, expected_namespace)

    def test_generate_subparser_with_only_interactive_short_flag(self):
        """Test generate subparser with only the short interactive flag."""
        parser = setup_rsmcmd_parser("test")
        parsed_namespace = parser.parse_args("generate -i".split())
        expected_namespace = argparse.Namespace(
            subcommand="generate", interactive=True, quiet=False
        )
        eq_(parsed_namespace, expected_namespace)

    def test_generate_subparser_with_subgroups_and_interactive_flags(self):
        """Test generate subparser with subgroups and interactive flags."""
        parser = setup_rsmcmd_parser("test", uses_subgroups=True)
        parsed_namespace = parser.parse_args("generate --interactive --subgroups".split())
        expected_namespace = argparse.Namespace(
            subcommand="generate", quiet=False, interactive=True, subgroups=True
        )
        eq_(parsed_namespace, expected_namespace)

    def test_generate_subparser_with_subgroups_and_interactive_short_flags(self):
        """Test generate subparser with short subgroups and interactive flags."""
        parser = setup_rsmcmd_parser("test", uses_subgroups=True)
        parsed_namespace = parser.parse_args("generate -i -g".split())
        expected_namespace = argparse.Namespace(
            subcommand="generate", quiet=False, interactive=True, subgroups=True
        )
        eq_(parsed_namespace, expected_namespace)

    def test_generate_subparser_with_subgroups_and_interactive_short_flags_together(
        self,
    ):
        """Test generate subparser with short subgroups and interactive flags together."""
        parser = setup_rsmcmd_parser("test", uses_subgroups=True)
        parsed_namespace = parser.parse_args("generate -ig".split())
        expected_namespace = argparse.Namespace(
            subcommand="generate", quiet=False, interactive=True, subgroups=True
        )
        eq_(parsed_namespace, expected_namespace)


class TestBatchGenerateConfiguration:
    @classmethod
    def setUpClass(cls):
        cls.expected_json_dir = join(rsmtool_test_dir, "data", "output")

    # a helper method to check that the automatically generated configuration
    # matches what we expect for each tool
    def check_generated_configuration(
        self, context, use_subgroups=False, as_string=False, suppress_warnings=False
    ):
        generator = ConfigurationGenerator(
            context,
            use_subgroups=use_subgroups,
            as_string=as_string,
            suppress_warnings=suppress_warnings,
        )

        if context == "rsmtool":
            configdict = {
                "experiment_id": "ENTER_VALUE_HERE",
                "model": "ENTER_VALUE_HERE",
                "train_file": "ENTER_VALUE_HERE",
                "test_file": "ENTER_VALUE_HERE",
            }

            if use_subgroups:
                section_list = [
                    "data_description",
                    "data_description_by_group",
                    "feature_descriptives",
                    "features_by_group",
                    "preprocessed_features",
                    "dff_by_group",
                    "consistency",
                    "model",
                    "evaluation",
                    "true_score_evaluation",
                    "evaluation_by_group",
                    "fairness_analyses",
                    "pca",
                    "intermediate_file_paths",
                    "sysinfo",
                ]
            else:
                section_list = [
                    "data_description",
                    "feature_descriptives",
                    "preprocessed_features",
                    "consistency",
                    "model",
                    "evaluation",
                    "true_score_evaluation",
                    "pca",
                    "intermediate_file_paths",
                    "sysinfo",
                ]

        elif context == "rsmeval":
            configdict = {
                "experiment_id": "ENTER_VALUE_HERE",
                "predictions_file": "ENTER_VALUE_HERE",
                "system_score_column": "ENTER_VALUE_HERE",
                "trim_min": "ENTER_VALUE_HERE",
                "trim_max": "ENTER_VALUE_HERE",
            }

            if use_subgroups:
                section_list = [
                    "data_description",
                    "data_description_by_group",
                    "consistency",
                    "evaluation",
                    "true_score_evaluation",
                    "evaluation_by_group",
                    "fairness_analyses",
                    "intermediate_file_paths",
                    "sysinfo",
                ]
            else:
                section_list = [
                    "data_description",
                    "consistency",
                    "evaluation",
                    "true_score_evaluation",
                    "intermediate_file_paths",
                    "sysinfo",
                ]

        elif context == "rsmcompare":
            configdict = {
                "comparison_id": "ENTER_VALUE_HERE",
                "experiment_id_old": "ENTER_VALUE_HERE",
                "experiment_dir_old": "ENTER_VALUE_HERE",
                "experiment_id_new": "ENTER_VALUE_HERE",
                "experiment_dir_new": "ENTER_VALUE_HERE",
                "description_old": "ENTER_VALUE_HERE",
                "description_new": "ENTER_VALUE_HERE",
            }

            if use_subgroups:
                section_list = [
                    "feature_descriptives",
                    "features_by_group",
                    "preprocessed_features",
                    "preprocessed_features_by_group",
                    "consistency",
                    "score_distributions",
                    "model",
                    "evaluation",
                    "true_score_evaluation",
                    "pca",
                    "notes",
                    "sysinfo",
                ]
            else:
                section_list = [
                    "feature_descriptives",
                    "preprocessed_features",
                    "consistency",
                    "score_distributions",
                    "model",
                    "evaluation",
                    "true_score_evaluation",
                    "pca",
                    "notes",
                    "sysinfo",
                ]

        elif context == "rsmsummarize":
            configdict = {
                "summary_id": "ENTER_VALUE_HERE",
                "experiment_dirs": ["ENTER_VALUE_HERE"],
            }

            section_list = [
                "preprocessed_features",
                "model",
                "evaluation",
                "true_score_evaluation",
                "intermediate_file_paths",
                "sysinfo",
            ]

        elif context == "rsmpredict":
            configdict = {
                "experiment_id": "ENTER_VALUE_HERE",
                "experiment_dir": "ENTER_VALUE_HERE",
                "input_features_file": "ENTER_VALUE_HERE",
            }

        elif context == "rsmexplain":
            configdict = {
                "experiment_id": "ENTER_VALUE_HERE",
                "background_data": "ENTER_VALUE_HERE",
                "explain_data": "ENTER_VALUE_HERE",
                "experiment_dir": "ENTER_VALUE_HERE",
            }

            section_list = [
                "data_description",
                "shap_values",
                "shap_plots",
            ]

        # get the generated configuration dictionary
        generated_configuration = generator.generate()

        # if we are testing string output, then load the expected json file
        # and compare its contents directly to the returned string, otherwise
        # compare the `_config` dictionaries of the two Configuration objects
        if as_string:
            if use_subgroups:
                expected_json_file = join(
                    self.expected_json_dir,
                    f"autogenerated_{context}_config_groups.json",
                )
            else:
                expected_json_file = join(
                    self.expected_json_dir, f"autogenerated_{context}_config.json"
                )
            expected_json_string = open(expected_json_file, "r").read().strip()
            eq_(generated_configuration, expected_json_string)
        else:
            expected_configuration_object = Configuration(configdict, context=context)
            if "general_sections" in expected_configuration_object:
                expected_configuration_object["general_sections"] = section_list

            assert_dict_equal(expected_configuration_object._config, generated_configuration)

    def test_generate_configuration(self):
        for context, use_subgroups, as_string, suppress_warnings in product(
            ["rsmtool", "rsmeval", "rsmcompare", "rsmsummarize", "rsmpredict"],
            [True, False],
            [True, False],
            [True, False],
        ):
            # rsmpredict, rsmsummarize and rsmexplain do not use subgroups
            if context in ["rsmpredict", "rsmsummarize", "rsmexplain"] and use_subgroups:
                continue

            yield (
                self.check_generated_configuration,
                context,
                use_subgroups,
                as_string,
                suppress_warnings,
            )


class TestInteractiveField:
    # class to test the InteractiveField class; note that we
    # are mocking the prompt_toolkit functionality
    # as per unit test best practices that recommend
    # not testing third-party libraries that are already
    # tested externally

    def check_boolean_field(self, field_type, user_input, final_value):
        """Check that boolean fields are handled correctly."""
        with patch("rsmtool.utils.commandline.prompt", return_value=user_input) as mock_prompt:
            ifield = InteractiveField(
                "test_bool",
                field_type,
                {"label": "answer the question", "type": "boolean"},
            )
            eq_(ifield.get_value(), final_value)
            eq_(mock_prompt.call_count, 1)
            eq_(
                mock_prompt.call_args[0][0].value,
                HTML(" <b>answer the question</b>: ").value,
            )

            # make sure the completer is set up correctly
            completer = mock_prompt.call_args[1]["completer"]
            eq_(completer.words, ["true", "false"])

            # make sure the validator validates the right things
            validator = mock_prompt.call_args[1]["validator"]
            eq_(validator.func("true"), True)
            eq_(validator.func("false"), True)
            eq_(validator.func(""), field_type == "optional")

            # boolean fields do not use a completion style
            complete_style = mock_prompt.call_args[1]["complete_style"]
            eq_(complete_style, None)

    def test_boolean_field(self):
        for field_type, (user_input, final_value) in product(
            ["required", "optional"], [("true", True), ("false", False)]
        ):
            yield self.check_boolean_field, field_type, user_input, final_value

    def check_choice_field(self, user_input, final_value):
        """Check that choice fields are handled correctly."""
        with patch("rsmtool.utils.commandline.prompt", return_value=user_input) as mock_prompt:
            ifield = InteractiveField(
                "test_choice",
                "required",
                {
                    "label": "pick a choice",
                    "choices": ["one", "two", "three"],
                    "type": "choice",
                },
            )
            eq_(ifield.get_value(), final_value)
            eq_(mock_prompt.call_count, 1)
            eq_(mock_prompt.call_args[0][0].value, HTML(" <b>pick a choice</b>: ").value)

            # make sure the completer is set up correctly
            completer = mock_prompt.call_args[1]["completer"]
            eq_(completer.words, ["one", "two", "three"])
            ok_(hasattr(completer, "fuzzy_completer"))

            # make sure the validator validates the right things
            validator = mock_prompt.call_args[1]["validator"]
            eq_(validator.func("one"), True)
            eq_(validator.func("two"), True)
            eq_(validator.func("three"), True)
            eq_(validator.func("four"), False)
            eq_(validator.func(""), False)

            # choice fields use multi-column completion style
            complete_style = mock_prompt.call_args[1]["complete_style"]
            eq_(complete_style, CompleteStyle.MULTI_COLUMN)

    def test_choice_field(self):
        for user_input, final_value in [("one", "one"), ("three", "three")]:
            yield self.check_choice_field, user_input, final_value

    @raises(ValueError)
    def test_choice_field_no_choices(self):
        _ = InteractiveField(
            "test_choice_no_choices",
            "required",
            {"label": "choose one", "type": "choice"},
        )

    def check_dir_field(self, user_input, final_value):
        """Check that dir fields are handled correctly."""
        with patch("rsmtool.utils.commandline.prompt", return_value=user_input) as mock_prompt:
            ifield = InteractiveField(
                "test_file", "required", {"label": "enter directory", "type": "dir"}
            )
            eq_(ifield.get_value(), final_value)
            eq_(mock_prompt.call_count, 1)
            eq_(
                mock_prompt.call_args[0][0].value,
                HTML(" <b>enter directory</b>: ").value,
            )

            # test that the path completer for files works as expected
            completer = mock_prompt.call_args[1]["completer"]
            eq_(completer.only_directories, True)
            eq_(completer.expanduser, False)

            # directories are okay
            eq_(completer.file_filter(rsmtool_test_dir), True)

            # make sure the validator validates the right things

            # create a temporary directory - it should work
            validator = mock_prompt.call_args[1]["validator"]
            tempdir = TemporaryDirectory()
            eq_(validator.func(tempdir.name), True)
            tempdir.cleanup()

            # other existing directories should also be accepted
            eq_(validator.func(rsmtool_test_dir), True)

            # but existing files should be rejected
            for extension in ["csv", "jsonlines", "sas7bdat", "tsv", "xlsx"]:
                existing_file = join(rsmtool_test_dir, "data", "files", f"train.{extension}")
                eq_(validator.func(existing_file), False)

            # file fields do not use a completion style
            complete_style = mock_prompt.call_args[1]["complete_style"]
            eq_(complete_style, None)

    def test_dir_field(self):
        for user_input, final_value in [("/foo/bar", "/foo/bar"), ("foo", "foo")]:
            yield self.check_dir_field, user_input, final_value

    def check_file_field(self, field_type, user_input, final_value):
        """Check that file fields are handled correctly."""
        with patch("rsmtool.utils.commandline.prompt", return_value=user_input) as mock_prompt:
            ifield = InteractiveField(
                "test_file", field_type, {"label": "enter file", "type": "file"}
            )
            eq_(ifield.get_value(), final_value)
            eq_(mock_prompt.call_count, 1)
            eq_(mock_prompt.call_args[0][0].value, HTML(" <b>enter file</b>: ").value)

            # test that the path completer for files works as expected
            completer = mock_prompt.call_args[1]["completer"]
            eq_(completer.only_directories, False)
            eq_(completer.expanduser, False)

            # directories are okay
            eq_(completer.file_filter(rsmtool_test_dir), True)

            # valid file formats are okay
            for extension in ["csv", "jsonlines", "sas7bdat", "tsv", "xlsx"]:
                valid_file = join(rsmtool_test_dir, "data", "files", f"train.{extension}")
                eq_(completer.file_filter(valid_file), True)

            eq_(
                completer.file_filter(
                    join(rsmtool_test_dir, "data", "experiments" "lr", "lr.json")
                ),
                False,
            )
            eq_(completer.file_filter(join(rsmtool_test_dir, "test_cli.py")), False)

            # make sure the validator validates the right things

            # create a temporary CSV file and close it right away
            # so that we know that it doesn't exist but has
            # the right extension
            validator = mock_prompt.call_args[1]["validator"]
            non_existing_csv_file = NamedTemporaryFile(suffix=".csv")
            non_existing_csv_file.close()
            eq_(validator.func(non_existing_csv_file.name), False)

            # directories should be rejected
            eq_(validator.func(rsmtool_test_dir), False)

            # existing files should be okay
            for extension in ["csv", "jsonlines", "sas7bdat", "tsv", "xlsx"]:
                existing_file = join(rsmtool_test_dir, "data", "files", f"train.{extension}")
                eq_(validator.func(existing_file), True)

            # empty files should are only accepted for optional fields
            eq_(validator.func(""), field_type == "optional")

            # file fields do not use a completion style
            complete_style = mock_prompt.call_args[1]["complete_style"]
            eq_(complete_style, None)

    def test_file_field(self):
        for field_type, (user_input, final_value) in product(
            ["required", "optional"], [("foo.csv", "foo.csv"), ("c.tsv", "c.tsv")]
        ):
            yield self.check_file_field, field_type, user_input, final_value

    def check_format_field(self, user_input, final_value):
        """Check that file format fields are handled correctly."""
        with patch("rsmtool.utils.commandline.prompt", return_value=user_input) as mock_prompt:
            ifield = InteractiveField(
                "test_file_format",
                "optional",
                {"label": "enter file format", "type": "format"},
            )
            eq_(ifield.get_value(), final_value)
            eq_(mock_prompt.call_count, 1)
            eq_(
                mock_prompt.call_args[0][0].value,
                HTML(" <b>enter file format</b>: ").value,
            )

            # test that the path completer for files works as expected
            completer = mock_prompt.call_args[1]["completer"]
            eq_(sorted(completer.words), ["csv", "tsv", "xlsx"])

            # make sure the validator validates the right things
            validator = mock_prompt.call_args[1]["validator"]
            eq_(validator.func("csv"), True)
            eq_(validator.func("CSV"), False)
            eq_(validator.func("tsv"), True)
            eq_(validator.func("TSV"), False)
            eq_(validator.func("xlsx"), True)
            eq_(validator.func("XLSX"), False)
            eq_(validator.func("xls"), False)
            eq_(validator.func(""), True)

            # file fields do not use a completion style
            complete_style = mock_prompt.call_args[1]["complete_style"]
            eq_(complete_style, None)

    def test_format_field(self):
        for user_input, final_value in [
            ("csv", "csv"),
            ("tsv", "tsv"),
            ("xlsx", "xlsx"),
        ]:
            yield self.check_format_field, user_input, final_value

    def check_id_field(self, user_input, final_value):
        """Check that ID fields are handled correctly."""
        with patch("rsmtool.utils.commandline.prompt", return_value=user_input) as mock_prompt:
            ifield = InteractiveField(
                "test_int", "required", {"label": "enter experiment ID", "type": "id"}
            )
            eq_(ifield.get_value(), final_value)
            eq_(mock_prompt.call_count, 1)
            eq_(
                mock_prompt.call_args[0][0].value,
                HTML(" <b>enter experiment ID</b>: ").value,
            )

            # there is no completer for integer field
            completer = mock_prompt.call_args[1]["completer"]
            eq_(completer, None)

            # make sure the validator validates the right things
            validator = mock_prompt.call_args[1]["validator"]
            eq_(validator.func("test"), True)
            eq_(validator.func("foo_bar"), True)
            eq_(validator.func("foo bar"), False)
            eq_(validator.func(""), False)

            # integer fields do not use a completion style
            complete_style = mock_prompt.call_args[1]["complete_style"]
            eq_(complete_style, None)

    def test_id_field(self):
        for user_input, final_value in [
            ("test", "test"),
            ("another_id", "another_id"),
        ]:
            yield self.check_id_field, user_input, final_value

    def check_integer_field(self, field_type, user_input, final_value):
        """Check that integer fields are handled correctly."""
        with patch("rsmtool.utils.commandline.prompt", return_value=user_input) as mock_prompt:
            ifield = InteractiveField(
                "test_int", field_type, {"label": "enter a number", "type": "integer"}
            )
            eq_(ifield.get_value(), final_value)
            eq_(mock_prompt.call_count, 1)
            eq_(
                mock_prompt.call_args[0][0].value,
                HTML(" <b>enter a number</b>: ").value,
            )

            # there is no completer for integer field
            completer = mock_prompt.call_args[1]["completer"]
            eq_(completer, None)

            # make sure the validator validates the right things;
            # recall the vaidator function for integer uses
            # regular expression matching
            validator = mock_prompt.call_args[1]["validator"]
            eq_(validator.func("5") is not None, True)
            eq_(validator.func("10") is not None, True)
            eq_(validator.func("9.5") is None, True)
            eq_(validator.func("abc") is None, True)
            eq_(validator.func("0") is not None, True)
            eq_(validator.func("") is not None, field_type == "optional")

            # integer fields do not use a completion style
            complete_style = mock_prompt.call_args[1]["complete_style"]
            eq_(complete_style, None)

    def test_integer_field(self):
        for (
            field_type,
            (user_input, final_value),
        ) in product(["required", "optional"], [("1", 1), ("10", 10), ("0", 0)]):
            yield self.check_integer_field, field_type, user_input, final_value

    def check_text_field(self, field_type, user_input, final_value):
        """Check that text fields are handled correctly."""
        with patch("rsmtool.utils.commandline.prompt", return_value=user_input) as mock_prompt:
            ifield = InteractiveField(
                "test_text", field_type, {"label": "description", "type": "text"}
            )
            eq_(ifield.get_value(), final_value)
            eq_(mock_prompt.call_count, 1)
            eq_(mock_prompt.call_args[0][0].value, HTML(" <b>description</b>: ").value)
            eq_(
                mock_prompt.call_args[1],
                {"completer": None, "complete_style": None, "validator": None},
            )

    def test_text_field(self):
        for field_type, (user_input, final_value) in product(
            ["required", "optional"], [("test", "test"), ("test value", "test value")]
        ):
            yield self.check_text_field, field_type, user_input, final_value

    def check_multiple_count_field(self, user_input, final_value, num_entries, field_type):
        """Check that fields that accept multiple values are handled correctly."""
        # int this particular case, we also need to patch the
        # `print_formatted_text()` function since we use that
        # for multiple count fields to print out the label, instead
        # of the `prompt()` function
        patcher = patch("rsmtool.utils.commandline.print_formatted_text")
        mock_print_formatted_text = patcher.start()

        with patch("rsmtool.utils.commandline.prompt", return_value=num_entries) as mock_prompt:
            ifield = InteractiveField(
                "test_multiple",
                "optional",
                {"label": "label for field", "count": "multiple", "type": field_type},
            )
            _ = ifield.get_value()
            # there are N + 1 calls to `prompt()`, one to get the number of entries
            # and one for each entry
            eq_(mock_prompt.call_count, int(num_entries) + 1)

            # check the first call to prompt that asks for the number of entries
            call_args = mock_prompt.call_args_list[0]
            eq_(call_args[0][0], "  How many do you want to specify: ")

            # this first prompt call has no completer and completion style
            # but it has a validator that only accepts integers including 0
            # but no blanks
            validator = call_args[1]["validator"]
            eq_(validator.func("5") is not None, True)
            eq_(validator.func("10") is not None, True)
            eq_(validator.func("9.5") is None, True)
            eq_(validator.func("abc") is None, True)
            eq_(validator.func("0") is not None, True)
            eq_(validator.func("") is None, True)

            # now let's check the subsequent calls to `prompt()`
            if num_entries != 0:
                for entry in range(1, int(num_entries) + 1):
                    call_args = mock_prompt.call_args_list[entry]
                    # check the label
                    eq_(call_args[0][0], f"   Enter #{entry}: ")

                    # now check the completers etc. depending on field type
                    if field_type == "text":
                        # for text fields, everythign is None
                        eq_(
                            call_args[1],
                            {
                                "completer": None,
                                "complete_style": None,
                                "validator": None,
                            },
                        )
                    else:
                        # for the dir type, the completer only accepts directories
                        completer = call_args[1]["completer"]
                        eq_(completer.only_directories, True)
                        eq_(completer.expanduser, False)

                        # and the validator only accepts real directories
                        validator = call_args[1]["validator"]
                        tempdir = TemporaryDirectory()
                        eq_(validator.func(tempdir.name), True)
                        tempdir.cleanup()

                        eq_(validator.func(rsmtool_test_dir), True)

                        # but existing files should be rejected
                        for extension in [
                            "csv",
                            "jsonlines",
                            "sas7bdat",
                            "tsv",
                            "xlsx",
                        ]:
                            existing_file = join(
                                rsmtool_test_dir, "data", "files", f"train.{extension}"
                            )
                            eq_(validator.func(existing_file), False)

                        complete_style = call_args[1]["complete_style"]
                        eq_(complete_style, None)

        # stop the `print_formatted_text()` patcher since we are done
        mock_print_formatted_text.stop()

    def test_multiple_count_field(self):
        for (user_input, final_value), num_entries, field_type in product(
            [("test", "test"), ("test value", "test value")],
            ["0", "3"],
            ["dir", "text"],
        ):
            yield (
                self.check_multiple_count_field,
                user_input,
                final_value,
                num_entries,
                field_type,
            )

    def check_optional_interactive_fields_blanks(self, field_name, field_count):
        """
        Check that blank user input for an optional field is handled correctly
        """
        default_value = DEFAULTS.get(field_name)
        blank_return_value = "" if field_count == "single" else []
        with patch("rsmtool.utils.commandline.prompt", return_value=blank_return_value):
            ifield = InteractiveField(field_name, "optional", {"label": "optional field label"})
            eq_(ifield.get_value(), default_value)

    def test_optional_interactive_fields_blanks(self):
        ALL_REQUIRED_FIELDS = set()
        for context in [
            "rsmtool",
            "rsmeval",
            "rsmpredict",
            "rsmsummarize",
            "rsmcompare",
            "rsmexplain",
        ]:
            ALL_REQUIRED_FIELDS.update(CHECK_FIELDS[context]["required"])
        OPTIONAL_INTERACTIVE_FIELDS = [
            field for field in INTERACTIVE_MODE_METADATA if field not in ALL_REQUIRED_FIELDS
        ]
        for field_name in OPTIONAL_INTERACTIVE_FIELDS:
            field_count = INTERACTIVE_MODE_METADATA[field_name].get("count", "single")
            yield self.check_optional_interactive_fields_blanks, field_name, field_count


class TestInteractiveGenerate:
    # class to test the interactive generation ; note that we
    # are mocking the prompt_toolkit functionality
    # as per unit test best practices that recommend
    # not testing third-party libraries that are already
    # tested externally

    # this class works by having a list of mocked up values
    # for each tool for the set of interactive fields that would
    # have been displayed (note that order matters) - this is
    # accomplished by using the `side_effect` functionality of
    # `patch()` that automatically iterates over the list of values

    # the actual test calls `ConfigurationGenerator.interact()` with
    # these mocked up values, generates a configuration and then
    # compares that configuration to the configuration we expect
    # given those values - that are stored in pre-computed JSON files

    # note that we are testing the interactive mode here rather than
    # in `test_cli.py` since it is not possible to mock things over
    # subprocess calls which is what `test_cli.py` uses for the `run`
    # subcommand and the non-interactive `generate` subcommand.

    @classmethod
    def setUpClass(cls):
        # define lists of mocked up values for each tool in the same order
        # that the interactive fields would have been displayed
        cls.mocked_rsmtool_interactive_values = [
            "testtool",  # experiment_id
            "Lasso",  # model
            "train.csv",  # train_file
            "test.csv",  # test_file
            "an rsmtool test",  # description
            False,  # exclude_zero_scores
            "csv",  # file_format
            "ID",  # id_column
            None,  # length_column
            "score2",  # second_human_score_column
            False,  # standardize_features
            ["L1", "QUESTION"],  # subgroups
            "score",  # test_label_column
            "score",  # train_label_column
            1,  # trim_min
            5,  # trim_max,
            True,  # use_scaled_predictions
            False,  # use_thumbnails
        ]

        cls.mocked_rsmeval_interactive_values = [
            "testeval",  # experiment_id
            "preds.csv",  # predictions_file
            "pred",  # system_score_column
            1,  # trim_min
            6,  # trim_max
            "an rsmeval test",  # description
            True,  # exclude_zeros
            "xlsx",  # file_format
            "score",  # human_score_column
            "ID",  # id_column
            "score2",  # second_human_score_column
            ["L1"],  # subgroups
            True,  # use_thumbnails
        ]

        cls.mocked_rsmcompare_interactive_values = [
            "testcompare",  # comparison_id
            "rsmtool1",  # experiment_id_old
            "/a/b/c",  # experiment_dir_old
            "rsmtool2",  # experiment_id_new
            "/d/e",  # experiment_dir_new
            "rsmtool expt 1",  # description_old
            "rsmtool expt 2",  # description_new
            [],  # subgroups
            True,  # use_thumbnails
        ]

        cls.mocked_rsmpredict_interactive_values = [
            "testpred",  # experiment_id
            "/a/b",  # experiment_dir_new
            "features.csv",  # input_features_file
            "csv",  # file_format
            "score",  # human_score_column
            "spkitemid",  # id_column
            None,  # second_human_score_column
            True,  # standardize_features
        ]

        cls.mocked_rsmsummarize_interactive_values = [
            "testsumm",  # summary_id
            ["/a/b", "/d", "/e/f/g"],  # experiment_dirs
            "summary test",  # description
            "tsv",  # file_format
            True,  # use_thumbnails
        ]

        cls.mocked_rsmxval_interactive_values = [
            "testxval",  # experiment_id
            "LinearSVC",  # model
            "train.csv",  # train_file
            "xval test",  # description
            True,  # exclude_zero_scores
            "xlsx",  # file_format
            3,  # folds
            None,  # folds file
            "ID",  # id_column
            "length",  # length_column
            "score2",  # second_human_score_column
            True,  # standardize_features
            ["L1", "QUESTION"],  # subgroups
            "score",  # train_label_column
            1,  # trim_min
            5,  # trim_max,
            True,  # use_scaled_predictions
            False,  # use_thumbnails
        ]

        cls.mocked_rsmxval_interactive_values_folds_file = [
            "testxval",  # experiment_id
            "LinearSVC",  # model
            "train.csv",  # train_file
            "xval test",  # description
            True,  # exclude_zero_scores
            "xlsx",  # file_format
            5,  # default folds
            "folds.csv",  # folds file
            "ID",  # id_column
            "length",  # length_column
            "score2",  # second_human_score_column
            True,  # standardize_features
            ["L1"],  # subgroups
            "score",  # train_label_column
            1,  # trim_min
            5,  # trim_max,
            True,  # use_scaled_predictions
            False,  # use_thumbnails
        ]

        cls.mocked_rsmexplain_interactive_values = [
            "train.csv",  # background_file
            "test.csv",  # explanable_file
            "testexplain",  # experiment_id
            "/a/b",  # RSMTool experiment_dir
            500,  # size of k-means sample for background
            "explain test",  # description
            "ID",  # id_column
            15,  # Number of features to be displayed
            None,  # Range of specific row IDs
            None,  # Size of random sample
            False,  # Show auto cohorts plot
            True,  # Standardize all features
        ]

    def check_tool_interact(self, context, subgroups=False, with_folds_file=False):
        """
        A helper method that runs `ConfigurationGenerator.interact()`
        and compares its output to expected output.

        Parameters
        ----------
        context : str
            Name of the tool being tested.
            One of {"rsmtool", "rsmeval", "rsmcompare", "rsmpredict", "rsmsummarize", "rsmexplain"}.
        subgroups : bool, optional
            Whether to include subgroup information in the generated configuration.
        with_folds_file : bool, optional
            Whether to use "folds_file" for rsmxval.
        """
        # if we are using subgroups, then define a suffix for the expected file
        groups_suffix = "_groups" if subgroups else ""
        folds_file_suffix = "_folds_file" if with_folds_file else ""

        # get the appropriate list of mocked values for this tool but make
        # a copy since we may need to modify it below
        mocked_values = getattr(self, f"mocked_{context}_interactive_values{folds_file_suffix}")[:]

        # if we are not using subgroups, delete the subgroup entry
        # from the list of mocked values
        if not subgroups:
            if context in ["rsmtool", "rsmeval"]:
                del mocked_values[11]
            elif context == "rsmcompare":
                del mocked_values[7]
            elif context == "rsmxval":
                del mocked_values[12]

        # point to the right file holding the expected configuration
        expected_file = f"interactive_{context}_config{groups_suffix}{folds_file_suffix}.json"
        expected_path = join(rsmtool_test_dir, "data", "output", expected_file)

        # we need to patch stderr and `prompt_toolkit.shortcuts.clear()`` so
        # that calling 'interact()' doesn't actually print out anything
        # to stderr and doesn't clear the screen
        sys_stderr_patcher = patch("sys.stderr", new_callable=StringIO)
        clear_patcher = patch("rsmtool.utils.commandline.clear")
        _ = clear_patcher.start()
        _ = sys_stderr_patcher.start()

        # mock the `InteractiveField.get_value()` method to return the
        # pre-determined mocked values in order and check that the
        # configuration generated by `interact()` is what we expect it to be
        with patch.object(InteractiveField, "get_value", side_effect=mocked_values):
            generator = ConfigurationGenerator(context, use_subgroups=subgroups)
            configuration_string = generator.interact()
            with open(expected_path, "r") as expectedfh:
                expected_configuration_string = expectedfh.read().strip()
                eq_(expected_configuration_string, configuration_string)

        # stop the stderr and clear patchers now that the test is finished
        sys_stderr_patcher.stop()
        clear_patcher.stop()

    def test_interactive_generate(self):
        # all tools except rsmpredict, rsmsummarize and rsmexplain
        # explicitly support subgroups; only rsmxval supports
        # folds file
        yield self.check_tool_interact, "rsmtool", False, False
        yield self.check_tool_interact, "rsmtool", True, False

        yield self.check_tool_interact, "rsmeval", False, False
        yield self.check_tool_interact, "rsmeval", True, False

        yield self.check_tool_interact, "rsmcompare", False, False
        yield self.check_tool_interact, "rsmcompare", True, False

        yield self.check_tool_interact, "rsmpredict", False, False

        yield self.check_tool_interact, "rsmsummarize", False

        yield self.check_tool_interact, "rsmexplain", False, False

        yield self.check_tool_interact, "rsmxval", False, False
        yield self.check_tool_interact, "rsmxval", False, True
        yield self.check_tool_interact, "rsmxval", True, False
        yield self.check_tool_interact, "rsmxval", True, True
