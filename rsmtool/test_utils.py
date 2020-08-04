import os
import re
import sys
import warnings

import numpy as np

from ast import literal_eval as eval
from bs4 import BeautifulSoup
from filecmp import clear_cache, dircmp
from glob import glob
from importlib.machinery import SourceFileLoader
from inspect import getmembers, getsourcelines, isfunction
from os import remove
from os.path import basename, exists, join
from pathlib import Path
from shutil import copyfile, copytree

from nose.tools import assert_equal, ok_
from pandas.testing import assert_frame_equal

from .reader import DataReader
from .modeler import Modeler
from .rsmtool import run_experiment
from .rsmcompare import run_comparison
from .rsmeval import run_evaluation
from .rsmpredict import compute_and_save_predictions
from .rsmsummarize import run_summary

html_error_regexp = re.compile(r'Traceback \(most recent call last\)')
html_warning_regexp = re.compile(r'<div class=".*?output_stderr.*?>')
section_regexp = re.compile(r'<h2>(.*?)</h2>')

# get the directory containing the tests
rsmtool_test_dir = Path(__file__).absolute().parent.parent.joinpath('tests')

tools_with_input_data = ['rsmsummarize', 'rsmcompare']
tools_with_output = ['rsmtool', 'rsmeval',
                     'rsmsummarize', 'rsmpredict']

# check if tests are being run in strict mode
# if so, any deprecation warnings found in HTML
# reports should not be ignored
STRICT_MODE = os.environ.get('STRICT', None)
IGNORE_DEPRECATION_WARNINGS = False if STRICT_MODE else True


def check_run_experiment(source,
                         experiment_id,
                         subgroups=None,
                         consistency=False,
                         skll=False,
                         file_format='csv',
                         given_test_dir=None,
                         config_obj_or_dict=None,
                         suppress_warnings_for=[]):
    """
    Function to run for a parameterized rsmtool experiment test.

    Parameters
    ----------
    source : str
        The name of the source directory containing the experiment
        configuration.
    experiment_id : str
        The experiment ID of the experiment.
    subgroups : list of str, optional
        List of subgroup names used in the experiment. If specified,
        outputs pertaining to subgroups are also checked as part of the
        test.
    consistency : bool, optional
        Whether to check consistency files as part of the experiment test.
        Generally, this should be true if the second human score column is
        specified. Defaults to `False`.
    skll : bool, optional
        Whether the model being used in the experiment is a SKLL model
        in which case the coefficients, predictions, etc. will not be
        checked since they can vary across machines, due to parameter tuning.
        Defaults to `False`.
    file_format : str, optional
        Which file format is being used for the output files of the experiment.
        Defaults to 'csv'.
    given_test_dir : str, optional
        Path where the test experiments are located. Unless specified, the
        rsmtool test directory is used. This can be useful when using these
        experiments to run tests for RSMExtra.
    config_obj_or_dict: Configuration or dictionary
        Configuration object or dictionary to use as an input.
        If None, the function will construct a path to the config file
        using `source` and `experiment_id`.
    suppress_warnings_for : list, optional
        Categories for which warnings should be suppressed when running the
        experiments.
    """
    # use the test directory from this file unless it's been overridden
    test_dir = given_test_dir if given_test_dir else rsmtool_test_dir

    if config_obj_or_dict is None:
        config_input = join(test_dir,
                            'data',
                            'experiments',
                            source,
                            '{}.json'.format(experiment_id))
    else:
        config_input = config_obj_or_dict

    model_type = 'skll' if skll else 'rsmtool'

    do_run_experiment(source,
                      experiment_id,
                      config_input,
                      suppress_warnings_for=suppress_warnings_for)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    output_files = glob(join(output_dir, '*.{}'.format(file_format)))
    for output_file in output_files:
        output_filename = basename(output_file)
        expected_output_file = join(expected_output_dir, output_filename)

        if exists(expected_output_file):
            check_file_output(output_file, expected_output_file, file_format=file_format)

    check_generated_output(output_files, experiment_id, model_type, file_format=file_format)

    if not skll:
        check_scaled_coefficients(source, experiment_id, file_format=file_format)

    if subgroups:
        check_subgroup_outputs(output_dir, experiment_id, subgroups, file_format=file_format)

    if consistency:
        check_consistency_files_exist(output_files, experiment_id, file_format=file_format)

    # check report for any errors but ignore warnings
    # which we check below separately
    check_report(html_report, raise_warnings=False)

    # make sure that there are no warnings in the report
    # but ignore deprecation warnings if appropriate
    warning_msgs = collect_warning_messages_from_report(html_report)
    if IGNORE_DEPRECATION_WARNINGS:
        warning_msgs = [msg for msg in warning_msgs if 'DeprecationWarning' not in msg]
    assert_equal(len(warning_msgs), 0)


def check_run_evaluation(source,
                         experiment_id,
                         subgroups=None,
                         consistency=False,
                         file_format='csv',
                         config_obj_or_dict=None,
                         given_test_dir=None,
                         suppress_warnings_for=[]):
    """
    Function to run for a parameterized rsmeval experiment test.

    Parameters
    ----------
    source : str
        The name of the source directory containing the experiment
        configuration.
    experiment_id : str
        The experiment ID of the experiment.
    subgroups : list of str, optional
        List of subgroup names used in the experiment. If specified,
        outputs pertaining to subgroups are also checked as part of the
        test.
    consistency : bool, optional
        Whether to check consistency files as part of the experiment test.
        Generally, this should be true if the second human score column is
        specified. Defaults to `False`.
    file_format : str, optional
        Which file format is being used for the output files of the experiment.
        Defaults to 'csv'.
    given_test_dir : str, optional
        Path where the test experiments are located. Unless specified, the
        rsmtool test directory is used. This can be useful when using these
        experiments to run tests for RSMExtra.
    config_obj_or_dict: Configuration or dict
        Configuration object or dictionary to use as an input.
        If None, the function will construct a path to the config file
        using ``source`` and ``experiment_id``.
    suppress_warnings_for : list, optional
        Categories for which warnings should be suppressed when running the
        experiments.
    """
    # use the test directory from this file unless it's been overridden
    test_dir = given_test_dir if given_test_dir else rsmtool_test_dir

    if config_obj_or_dict is None:
        config_input = join(test_dir,
                            'data',
                            'experiments',
                            source,
                            '{}.json'.format(experiment_id))
    else:
        config_input = config_obj_or_dict

    do_run_evaluation(source,
                      experiment_id,
                      config_input,
                      suppress_warnings_for=suppress_warnings_for)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    output_files = glob(join(output_dir, '*.{}'.format(file_format)))
    for output_file in output_files:
        output_filename = basename(output_file)
        expected_output_file = join(expected_output_dir, output_filename)

        if exists(expected_output_file):
            check_file_output(output_file, expected_output_file, file_format=file_format)

    if consistency:
        check_consistency_files_exist(output_files, experiment_id)

    # check report for any errors but ignore warnings
    # which we check below separately
    check_report(html_report, raise_warnings=False)

    # make sure that there are no warnings in the report
    # but ignore deprecation warnings if appropriate
    warning_msgs = collect_warning_messages_from_report(html_report)
    if IGNORE_DEPRECATION_WARNINGS:
        warning_msgs = [msg for msg in warning_msgs if 'DeprecationWarning' not in msg]
    assert_equal(len(warning_msgs), 0)


def check_run_comparison(source,
                         experiment_id,
                         given_test_dir=None,
                         config_obj_or_dict=None,
                         suppress_warnings_for=[]):
    """
    Function to run for a parameterized rsmcompare experiment test.

    Parameters
    ----------
    source : str
        The name of the source directory containing the experiment
        configuration.
    experiment_id : str
        The experiment ID of the experiment.
    given_test_dir : str, optional
        Path where the test experiments are located. Unless specified, the
        rsmtool test directory is used. This can be useful when using these
        experiments to run tests for RSMExtra.
    config_obj_or_dict: Configuration or dict
        Configuration object or dictionary to use as an input.
        If None, the function will construct a path to the config file
        using ``source`` and ``experiment_id``.
    suppress_warnings_for : list, optional
        Categories for which warnings should be suppressed when running the
        experiments.
    """
    # use the test directory from this file unless it's been overridden
    test_dir = given_test_dir if given_test_dir else rsmtool_test_dir

    if config_obj_or_dict is None:
        config_input = join(test_dir,
                            'data',
                            'experiments',
                            source,
                            'rsmcompare.json')
    else:
        config_input = config_obj_or_dict

    do_run_comparison(source,
                      config_input,
                      suppress_warnings_for=suppress_warnings_for)

    html_report = join('test_outputs', source, '{}_report.html'.format(experiment_id))

    # check report for any errors but ignore warnings
    # which we check below separately
    check_report(html_report, raise_warnings=False)

    # make sure that there are no warnings in the report
    # but ignore deprecation warnings if appropriate
    warning_msgs = collect_warning_messages_from_report(html_report)
    if IGNORE_DEPRECATION_WARNINGS:
        warning_msgs = [msg for msg in warning_msgs if 'DeprecationWarning' not in msg]
    assert_equal(len(warning_msgs), 0)


def check_run_prediction(source,
                         excluded=False,
                         file_format='csv',
                         given_test_dir=None,
                         config_obj_or_dict=None,
                         suppress_warnings_for=[]):
    """
    Function to run for a parameterized rsmpredict experiment test.

    Parameters
    ----------
    source : str
        The name of the source directory containing the experiment
        configuration.
    excluded : bool, optional
        Whether to check the excluded responses file as part of the test.
        Defaults to `False`.
    file_format : str, optional
        Which file format is being used for the output files of the experiment.
        Defaults to 'csv'.
    given_test_dir : str, optional
        Path where the test experiments are located. Unless specified, the
        rsmtool test directory is used. This can be useful when using these
        experiments to run tests for RSMExtra.
    config_obj_or_dict: Configuration or dict
        Configuration object or dictionary to use as an input.
        If None, the function will construct a path to the config file
        using ``source`` and ``experiment_id``.
    suppress_warnings_for : list, optional
        Categories for which warnings should be suppressed when running the
        experiments.
    """
    # use the test directory from this file unless it's been overridden
    test_dir = given_test_dir if given_test_dir else rsmtool_test_dir

    if config_obj_or_dict is None:
        config_input = join(test_dir,
                            'data',
                            'experiments',
                            source,
                            'rsmpredict.json')
    else:
        config_input = config_obj_or_dict

    do_run_prediction(source,
                      config_input,
                      suppress_warnings_for=suppress_warnings_for)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    output_files = ['predictions.{}'.format(file_format),
                    'preprocessed_features.{}'.format(file_format)]
    if excluded:
        output_files.append('predictions_excluded_responses.{}'.format(file_format))
    for output_file in output_files:
        generated_output_file = join(output_dir, output_file)
        expected_output_file = join(expected_output_dir, output_file)

        check_file_output(generated_output_file, expected_output_file)


def check_run_summary(source,
                      file_format='csv',
                      given_test_dir=None,
                      config_obj_or_dict=None,
                      suppress_warnings_for=[]):
    """
    Function to run for a parameterized rsmsummarize experiment test.

    Parameters
    ----------
    source : str
        The name of the source directory containing the experiment
        configuration.
    file_format : str, optional
        Which file format is being used for the output files of the experiment.
        Defaults to 'csv'.
    given_test_dir : str, optional
        Path where the test experiments are located. Unless specified, the
        rsmtool test directory is used. This can be useful when using these
        experiments to run tests for RSMExtra.
    config_obj_or_dict: Configuration or dict
        Configuration object or dictionary to use as an input.
        If None, the function will construct a path to the config file
        using ``source`` and ``experiment_id``.
    suppress_warnings_for : list, optional
        Categories for which warnings should be suppressed when running the
        experiments.
    """
    # use the test directory from this file unless it's been overridden
    test_dir = given_test_dir if given_test_dir else rsmtool_test_dir

    if config_obj_or_dict is None:
        config_input = join(test_dir,
                            'data',
                            'experiments',
                            source,
                            'rsmsummarize.json')
    else:
        config_input = config_obj_or_dict

    do_run_summary(source,
                   config_input,
                   suppress_warnings_for=suppress_warnings_for)

    html_report = join('test_outputs', source, 'report', 'model_comparison_report.html')

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    output_files = glob(join(output_dir, '*.{}'.format(file_format)))
    for output_file in output_files:
        output_filename = basename(output_file)
        expected_output_file = join(expected_output_dir, output_filename)

        if exists(expected_output_file):
            check_file_output(output_file, expected_output_file)

    # check report for any errors but ignore warnings
    # which we check below separately
    check_report(html_report, raise_warnings=False)

    # make sure that there are no warnings in the report
    # but ignore deprecation warnings if appropriate
    warning_msgs = collect_warning_messages_from_report(html_report)
    if IGNORE_DEPRECATION_WARNINGS:
        warning_msgs = [msg for msg in warning_msgs if 'DeprecationWarning' not in msg]
    assert_equal(len(warning_msgs), 0)


def do_run_experiment(source,
                      experiment_id,
                      config_input,
                      suppress_warnings_for=[]):
    """
    Run RSMTool experiment using the given experiment
    configuration file located in the given source directory
    and using the given experiment ID.

    Parameters
    ----------
    source : str
        Path to where the test is located on disk.
    experiment_id : str
        Experiment ID to use when running.
    config_input : str or Configuration or dict
        Path to the experiment configuration file,
        or a `Configuration` object
        or a Python dictionary with keys corresponding to fields in the
        configuration file.
    suppress_warnings_for : list, optional
        Categories for which warnings should be suppressed
        when running the experiments. Note that ``RuntimeWarning``s
        are always suppressed.
    """
    source_output_dir = 'test_outputs'
    experiment_dir = join(source_output_dir, source)

    # remove all previously created files
    for output_subdir in ['output', 'figure', 'report']:
        files = glob(join(source_output_dir, source, output_subdir, '*'))
        for f in files:
            remove(f)

    with warnings.catch_warnings():

        # always suppress runtime warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        # suppress additional warning types if specified
        for warning_type in suppress_warnings_for:
            warnings.filterwarnings('ignore', category=warning_type)

        run_experiment(config_input, experiment_dir)


def do_run_evaluation(source,
                      experiment_id,
                      config_input,
                      suppress_warnings_for=[]):
    """
    Run RSMEval experiment using the given experiment
    configuration file located in the given source directory
    and using the given experiment ID.

    Parameters
    ----------
    source : str
        Path to where the test is located on disk.
    experiment_id : str
        Experiment ID to use when running.
    config_input : str or Configuration or dict
        Path to the experiment configuration file,
        or a `Configuration` object
        or a Python dictionary with keys corresponding to fields in the
        configuration file.
    suppress_warnings_for : list, optional
        Categories for which warnings should be suppressed
        when running the experiments. Note that ``RuntimeWarning``s
        are always suppressed.
    """
    source_output_dir = 'test_outputs'
    experiment_dir = join(source_output_dir, source)

    # remove all previously created files
    for output_subdir in ['output', 'figure', 'report']:
        files = glob(join(source_output_dir, source, output_subdir, '*'))
        for f in files:
            remove(f)

    with warnings.catch_warnings():

        # always suppress runtime warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        # suppress additional warning types if specified
        for warning_type in suppress_warnings_for:
            warnings.filterwarnings('ignore', category=warning_type)

        run_evaluation(config_input, experiment_dir)


def do_run_prediction(source,
                      config_input,
                      suppress_warnings_for=[]):
    """
    Run RSMPredict experiment using the given experiment
    configuration file located in the given source directory.

    Parameters
    ----------
    source : str
        Path to where the test is located on disk.
    config_input : str or Configuration or dict
        Path to the experiment configuration file,
        or a `Configuration` object
        or a Python dictionary with keys corresponding to fields in the
        configuration file
    suppress_warnings_for : list, optional
        Categories for which warnings should be suppressed when running the
        experiments.
    """
    source_output_dir = 'test_outputs'

    # The `csv` file extension is ultimately dropped by the `rsmpredict.py`
    # script, so these arguments can be used for CSV, TSV, or XLSX output
    output_file = join(source_output_dir, source, 'output', 'predictions.csv')
    feats_file = join(source_output_dir, source, 'output', 'preprocessed_features.csv')

    # remove all previously created files
    files = glob(join(source_output_dir, 'output', '*'))
    for f in files:
        remove(f)

    with warnings.catch_warnings():

        # always suppress runtime warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        # suppress additional warning types if specified
        for warning_type in suppress_warnings_for:
            warnings.filterwarnings('ignore', category=warning_type)

        compute_and_save_predictions(config_input, output_file, feats_file)


def do_run_comparison(source,
                      config_input,
                      suppress_warnings_for=[]):
    """
    Run RSMCompare experiment using the given experiment
    configuration file located in the given source directory.

    Parameters
    ----------
    source : str
        Path to where the test is located on disk.
    config_input : str or Configuration or dict
        Path to the experiment configuration file,
        or a `Configuration` object
        or a Python dictionary with keys corresponding to fields in the
        configuration file
    suppress_warnings_for : list, optional
        Categories for which warnings should be suppressed
        when running the experiments. Note that ``RuntimeWarning``s
        are always suppressed.
    """
    source_output_dir = 'test_outputs'
    experiment_dir = join(source_output_dir, source)

    with warnings.catch_warnings():

        # always suppress runtime warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        # suppress additional warning types if specified
        for warning_type in suppress_warnings_for:
            warnings.filterwarnings('ignore', category=warning_type)

        run_comparison(config_input, experiment_dir)


def do_run_summary(source,
                   config_input,
                   suppress_warnings_for=[]):
    """
    Run rsmsummarizeary experiment using the given experiment
    configuration file located in the given source directory.

    Parameters
    ----------
    source : str
        Path to where the test is located on disk.
    config_input : str or Configuration or dict
        Path to the experiment configuration file,
        or a `Configuration` object
        or a Python dictionary with keys corresponding to fields in the
        configuration file
    suppress_warnings_for : list, optional
        Categories for which warnings should be suppressed when running the
        experiments.
    """
    source_output_dir = 'test_outputs'
    experiment_dir = join(source_output_dir, source)

    # remove all previously created files
    for output_subdir in ['output', 'figure', 'report']:
        files = glob(join(source_output_dir, source, output_subdir, '*'))
        for f in files:
            remove(f)

    with warnings.catch_warnings():

        # always suppress runtime warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        # suppress additional warning types if specified
        for warning_type in suppress_warnings_for:
            warnings.filterwarnings('ignore', category=warning_type)

        run_summary(config_input, experiment_dir)


def check_file_output(file1, file2, file_format='csv'):
    """
    Check if two experiment files have values that are
    the same to within three decimal places. Raises an
    AssertionError if they are not.

    Parameters
    ----------
    file1 : str
        Path to the first file.
    file2 : str
        Path to the second files.
    file_format : str, optional
        The format of the output files.
        Defaults to 'csv'.
    """

    # make sure that the main id columns are read as strings since
    # this may affect merging in custom notebooks
    string_columns = ['spkitemid', 'candidate']

    converter_dict = {column: str for column in string_columns}

    df1 = DataReader.read_from_file(file1, converters=converter_dict)
    df2 = DataReader.read_from_file(file2, converters=converter_dict)

    # convert all column names to strings
    # we do this to avoid any errors during sorting.
    for df in [df1, df2]:
        df.columns = df.columns.map(str)

    # if the first column is numeric, just force the index to string;
    # however, if it is non-numeric, assume that it is an index and
    # force it to string. We do this to ensure string indices are
    # preserved as such
    for df in [df1, df2]:
        if np.issubdtype(df[df.columns[0]].dtype, np.number):
            df.index = df.index.map(str)
        else:
            df.index = df[df.columns[0]]
            df.index = df.index.map(str)

    # sort all the indices alphabetically
    df1.sort_index(inplace=True)
    df2.sort_index(inplace=True)

    # sort all columns alphabetically
    df1.sort_index(axis=1, inplace=True)
    df2.sort_index(axis=1, inplace=True)

    # convert any integer columns to floats in either data frame
    for df in [df1, df2]:
        for c in df.columns:
            if df[c].dtype == np.int64:
                df[c] = df[c].astype(np.float64)

    # for pca and factor correlations convert all values to absolutes
    # because the sign may not always be the same
    if (file1.endswith('pca.{}'.format(file_format)) or
            file1.endswith('factor_correlations.{}'.format(file_format))):
        for df in [df1, df2]:
            msk = df.dtypes == np.float64
            df.loc[:, msk] = df.loc[:, msk].abs()

    try:
        assert_frame_equal(df1,
                           df2,
                           check_exact=False,
                           rtol=1e-03)
    except AssertionError as e:
        message = e.args[0]
        new_message = 'File {} - {}'.format(basename(file1), message)
        e.args = (new_message, )
        raise


def collect_warning_messages_from_report(html_file):
    """
    Collect the warning messages from HTML report file.

    Parameters
    ----------
    html_file : str
        Path the HTML report file on disk.

    Returns
    -------
    warnings_text : list of str
        The list of warnings
    """
    with open(html_file, 'r') as htmlf:
        soup = BeautifulSoup(htmlf.read(), 'html.parser')

    warnings_text = []
    for div in soup.findAll("div", {"class": "output_stderr"}):

        # we collect the text in the <pre> tags after the standard error,
        # and split the lines; we only keep the lines that contain 'Warning:'
        for pre in div.findAll("pre"):
            warnings_msgs = pre.text.splitlines()
            warnings_msgs = [msg for msg in warnings_msgs if 'Warning:' in msg]
            warnings_text.extend(warnings_msgs)

    return warnings_text


def check_report(html_file,
                 raise_errors=True,
                 raise_warnings=True):
    """
    Checks if the HTML report contains any errors.
    Raises an AssertionError if it does.

    Parameters
    ----------
    html_file : str
        Path the HTML report file on disk.
    raise_errors : bool, optional
        Whether to raise assertion error if there
        are any errors in the report.
        Defaults to True.
    raise_warnings : bool, optional
        Whether to raise assertion error if there
        are any warnings in the report.
        Defaults to True.
    """
    report_errors = 0
    report_warnings = 0

    with open(html_file, 'r') as htmlf:
        for line in htmlf:
            m_error = html_error_regexp.search(line)
            if m_error:
                report_errors += 1
            m_warning = html_warning_regexp.search(line)
            if m_warning:
                report_warnings += 1

    if raise_errors:
        assert_equal(report_errors, 0)
    if raise_warnings:
        assert_equal(report_warnings, 0)


def check_scaled_coefficients(source, experiment_id, file_format='csv'):
    """
    Check that the predictions generated using scaled
    coefficients match the scaled scores. Raises an
    AssertionError if they do not.

    Parameters
    ----------
    source : str
        Path to the source directory on disk.
    experiment_id : str
        The experiment ID.
    file_format : str, optional
        The format of the output files.
        Defaults to 'csv'.
    """
    preprocessed_test_file = join('test_outputs',
                                  source,
                                  'output',
                                  '{}_test_preprocessed_features.{}'.format(experiment_id,
                                                                            file_format))
    scaled_coefficients_file = join('test_outputs',
                                    source,
                                    'output',
                                    '{}_coefficients_scaled.{}'.format(experiment_id,
                                                                       file_format))
    predictions_file = join('test_outputs',
                            source,
                            'output',
                            '{}_pred_processed.{}'.format(experiment_id,
                                                          file_format))

    postprocessing_params_file = join('test_outputs',
                                      source,
                                      'output',
                                      '{}_postprocessing_params.{}'.format(experiment_id,
                                                                           file_format))

    postproc_params = DataReader.read_from_file(postprocessing_params_file).loc[0]
    df_preprocessed_test_data = DataReader.read_from_file(preprocessed_test_file)
    df_old_predictions = DataReader.read_from_file(predictions_file)
    df_old_predictions = df_old_predictions[['spkitemid', 'sc1', 'scale']]

    # create fake skll objects with new coefficients
    df_coef = DataReader.read_from_file(scaled_coefficients_file)
    learner = Modeler.create_fake_skll_learner(df_coef)
    modeler = Modeler.load_from_learner(learner)

    # generate new predictions and rename the prediction column to 'scale'
    df_new_predictions = modeler.predict(df_preprocessed_test_data,
                                         postproc_params['trim_min'],
                                         postproc_params['trim_max'])
    df_new_predictions.rename(columns={'raw': 'scale'}, inplace=True)

    # check that new predictions match the scaled old predictions
    assert_frame_equal(df_new_predictions.sort_index(axis=1),
                       df_old_predictions.sort_index(axis=1),
                       check_exact=False,
                       rtol=1e-03)


def check_generated_output(generated_files, experiment_id, model_source, file_format='csv'):
    """
    Check that all crucial output files have been generated.
    Raises an AssertionError if they have not.

    Parameters
    ----------
    generated_files : list of str
        List of files generated by a test.
    experiment_id : str
        The experiment ID.
    model_source : str
        'rsmtool' or 'skll'
    file_format : str, optional
        The format of the output files.
        Defaults to 'csv'.
    """
    file_must_have_both = ["_confMatrix.{}".format(file_format),
                           "_cors_orig.{}".format(file_format),
                           "_cors_processed.{}".format(file_format),
                           "_eval.{}".format(file_format),
                           "_eval_short.{}".format(file_format),
                           "_feature.{}".format(file_format),
                           "_feature_descriptives.{}".format(file_format),
                           "_feature_descriptivesExtra.{}".format(file_format),
                           "_feature_outliers.{}".format(file_format),
                           "_margcor_score_all_data.{}".format(file_format),
                           "_pca.{}".format(file_format),
                           "_pcavar.{}".format(file_format),
                           "_pcor_score_all_data.{}".format(file_format),
                           "_pred_processed.{}".format(file_format),
                           "_pred_train.{}".format(file_format),
                           "_score_dist.{}".format(file_format),
                           "_train_preprocessed_features.{}".format(file_format),
                           "_test_preprocessed_features.{}".format(file_format),
                           "_postprocessing_params.{}".format(file_format)
                           ]

    file_must_have_rsmtool = ["_betas.{}".format(file_format),
                              "_coefficients.{}".format(file_format)]
    if model_source == 'rsmtool':
        file_must_have = file_must_have_both + file_must_have_rsmtool
    else:
        file_must_have = file_must_have_both

    file_must_with_id = [experiment_id + file_name for file_name in file_must_have]
    file_exist = [basename(file_name) for file_name in generated_files]
    missing_file = set(file_must_with_id).difference(set(file_exist))
    assert_equal(len(missing_file), 0, "Missing files: {}".format(','.join(missing_file)))


def check_consistency_files_exist(generated_files, experiment_id, file_format='csv'):
    """
    Check to make sure that the consistency files
    were generated. Raises an AssertionError if
    they were not.

    Parameters
    ----------
    generated_files : list of str
        List of files generated by a test.
    experiment_id : str
        The experiment ID.
    file_format : str, optional
        The format of the output files.
        Defaults to 'csv'.
    """
    file_must_have = ["_consistency.{}".format(file_format),
                      "_degradation.{}".format(file_format),
                      "_disattenuated_correlations.{}".format(file_format),
                      "_true_score_eval.{}".format(file_format)]

    file_must_with_id = [experiment_id + file_name for file_name in file_must_have]
    file_exist = [basename(file_name) for file_name in generated_files]
    missing_file = set(file_must_with_id).difference(set(file_exist))
    assert_equal(len(missing_file), 0, "Missing files: {}".format(','.join(missing_file)))


def check_subgroup_outputs(output_dir, experiment_id, subgroups, file_format='csv'):
    """
    Check to make sure that the subgroup outputs
    look okay. Raise an AssertionError if they do not.

    Parameters
    ----------
    output_dir : str
        Path to the `output` experiment output directory for a test.
    experiment_id : str
        The experiment ID.
    subgroups : list of str
        List of column names that contain grouping
        information.
    file_format : str, optional
        The format of the output files.
        Defaults to 'csv'.
    """
    train_preprocessed_file = join(output_dir,
                                   '{}_train_metadata.{}'.format(experiment_id,
                                                                 file_format))
    train_preprocessed = DataReader.read_from_file(train_preprocessed_file, index_col=0)

    test_preprocessed_file = join(output_dir,
                                  '{}_test_metadata.{}'.format(experiment_id,
                                                               file_format))
    test_preprocessed = DataReader.read_from_file(test_preprocessed_file,
                                                  index_col=0)
    for group in subgroups:
        ok_(group in train_preprocessed.columns)
        ok_(group in test_preprocessed.columns)

    # check that the total sum of N per category matches the total N
    # in data composition and the total N categories matches what is
    # in overall data composition
    file_data_composition_all = join(output_dir,
                                     '{}_data_composition.{}'.format(experiment_id,
                                                                     file_format))
    df_data_composition_all = DataReader.read_from_file(file_data_composition_all)
    for group in subgroups:
        file_composition_by_group = join(output_dir,
                                         '{}_data_composition_by_{}.{}'.format(experiment_id,
                                                                               group,
                                                                               file_format))
        composition_by_group = DataReader.read_from_file(file_composition_by_group)
        for partition in ['Training', 'Evaluation']:
            partition_info = df_data_composition_all.loc[df_data_composition_all['partition'] ==
                                                         partition]

            summation = sum(composition_by_group['{} set'
                                                 ''.format(partition)])
            ok_(summation == partition_info.iloc[0]['responses'])

            length = len(composition_by_group.loc[composition_by_group['{} set'
                                                                       ''.format(partition)] != 0])
            ok_(length == partition_info.iloc[0][group])


def copy_data_files(temp_dir_name,
                    input_file_dict,
                    given_test_dir):
    """
    A utility function to copy files from the given test directory into
    a specified temporary directory. Useful for tests where the
    current directory is to be used as the reference for resolving paths
    in the configuration.

    Parameters
    ----------
    temp_dir_name : str
        Name of the temporary directory.
    input_file_dict : dict
        A dictionary of files/directories to copy with keys as the
        file type and the values are their paths relative to the `tests`
        directory.
    given_test_dir : str
        Directory where the the test experiments are located. This can be
        useful when using these experiments to run tests for RSMExtra.

    Returns
    -------
    output_file_dict : dict
        The dictionary with the same keys as
        input_file_dict and values showing new paths.
    """

    temp_dir = Path(temp_dir_name)
    if not temp_dir.exists():
        temp_dir.mkdir()

    output_file_dict = {}
    for file in input_file_dict:
        filepath = Path(input_file_dict[file])
        filename = filepath.name
        old_filepath = given_test_dir / filepath
        new_filepath = temp_dir / filename
        if old_filepath.is_dir():
            copytree(old_filepath, new_filepath)
        else:
            copyfile(old_filepath, new_filepath)
        output_file_dict[file] = str(new_filepath)

    return output_file_dict


class FileUpdater(object):

    """
    A FileUpdater object is used to update the test outputs
    for the tests in the `tests_directory` based on the
    outputs contained in the `updated_outputs_directory`.
    It does this for all of the experiment tests contained
    in the test files given by each of the `test_suffixes`.

    Attributes
    ----------
    test_suffixes : list
        List of suffixes that will be added to the string
        "test_experiment_" and located in the `tests_directory` to find
        the tests that are to be updated.
    tests_directory : str
        Path to the directory containing the tests whose outputs are
        to be updated.
    updated_outputs_directory : str
        Path to the directory containing the updated outputs for the
        experiment tests.
    deleted_files : list
        List of files deleted from `tests directory`.
    updated_files : list
        List of files that have either (really) changed in the updated outputs
        or been added in those outputs.
    missing_or_empty_sources : list
        List of source names whose corresponding directories are either
        missing under `updated_outputs_directory` or do exist but are
        empty.
    """

    def __init__(self,
                 test_suffixes,
                 tests_directory,
                 updated_outputs_directory):
        self.test_suffixes = test_suffixes
        self.tests_directory = Path(tests_directory)
        self.updated_outputs_directory = Path(updated_outputs_directory)
        self.missing_or_empty_sources = []
        self.deleted_files = []
        self.updated_files = []

        # invalidate the file comparison cache
        clear_cache()

    def is_skll_excluded_file(self, filename):
        """
        Checks whether the given filename is one that should
        be excluded for SKLL-based experiment tests.

        Parameters
        ----------
        filename : str
            Name of the file to be checked.

        Returns
        -------
        exclude
            `True` if the file should be excluded.
            `False` otherwise.
        """
        possible_suffixes = ['.model', '.npy']
        possible_stems = ['_postprocessing_params', '_eval', '_eval_short',
                          '_confMatrix', '_pred_train', '_pred_processed',
                          '_score_dist']

        file_stem = Path(filename).stem
        file_suffix = Path(filename).suffix
        return any(file_suffix == suffix for suffix in possible_suffixes) or \
            any(file_stem.endswith(stem) for stem in possible_stems)

    def update_source(self,
                      source,
                      skll=False,
                      file_type='output',
                      input_source=None):
        """
        Update the test output or input data for experiment test with `source` as the
        given name. It deletes files that are only in the tests directory,
        adds files that are only in the updated test outputs directory, and
        updates the files that have changed in the updated test outputs directory.
        It does not return anything but updates the `deleted_files`, `updated_files`,
        and `missing_or_empty_sources`  class attributes appropriately.

        Parameters
        ----------
        source : str
            Name of source directory.
        skll : bool, optional
            Whether the given source
            is for a SKLL-based experiment test.
        file_type: str, optional
            Whether we are updating test output files or test input files.
            Input files are updated for rsmtool and rsmcompare.
        input_source: str, optional
            The name of the source directory for input files
        """
        # locate the updated outputs for the experiment under the given
        # outputs directory, locate the existing experiment outputs
        # and define how we will refer to the test
        if file_type == 'output':
            updated_output_path = self.updated_outputs_directory / source / "output"
            existing_output_path = self.tests_directory / "data" / "experiments" / source / "output"
            test_name = source
        else:
            updated_output_path = self.updated_outputs_directory / input_source / "output"
            existing_output_path = (self.tests_directory / "data" / "experiments" / source /
                                    input_source / "output")
            test_name = f'{source}/{input_source}'

        # if the directory for this source does not exist on the updated output
        # side, then that's a problem and something we should report on later
        try:
            assert updated_output_path.exists()
        except AssertionError:
            self.missing_or_empty_sources.append(test_name)
            return

        # if the existing output path does not exist, then create it
        try:
            assert existing_output_path.exists()
        except AssertionError:
            sys.stderr.write("\nNo existing output for \"{}\". "
                             "Creating directory ...\n".format(test_name))
            existing_output_path.mkdir(parents=True)

        # get a comparison betwen the two directories
        dir_comparison = dircmp(updated_output_path, existing_output_path)

        # if no output was found in the updated outputs directory, that's
        # likely to be a problem so save that source
        if not dir_comparison.left_list:
            self.missing_or_empty_sources.append(test_name)
            return

        # first delete the files that only exist in the existing output directory
        # since those are likely old files from old versions that we do not need
        existing_output_only_files = dir_comparison.right_only
        for file in existing_output_only_files:
            remove(existing_output_path / file)

        # Next find all the NEW files in the updated outputs.
        new_files = dir_comparison.left_only

        # We also define several types of files we exclude.
        # 1. we exclude OLS summary files
        excluded_suffixes = ['_ols_summary.txt',
                             '.ols', '.model', '.npy']

        # 2. for output files we exclude all json files.
        # We keep these files if we are dealing with input files.
        if file_type == 'output':
            excluded_suffixes.extend(['_rsmtool.json', '_rsmeval.json',
                                      '_rsmsummarize.json', '_rsmcompare.json'])

        new_files = [f for f in new_files if not any(f.endswith(suffix) for suffix in excluded_suffixes)]

        # 3. We also exclude files related to model evaluations for SKLL models.
        if skll:
            new_files = [f for f in new_files if not self.is_skll_excluded_file(f)]

        # next we get the files that have changed and try to figure out if they
        # have actually changed beyond a tolerance level that we care about for
        # tests. To do this, we run the same function that we use when comparing
        # the files in the actual test. However, for non-tabular files, we just
        # assume that they have really changed since we have no easy way to compare.
        changed_files = dir_comparison.diff_files
        really_changed_files = []
        for changed_file in changed_files:
            include_file = True
            updated_output_filepath = updated_output_path / changed_file
            existing_output_filepath = existing_output_path / changed_file
            file_format = updated_output_filepath.suffix.lstrip('.')
            if file_format in ['csv', 'tsv', 'xlsx']:
                try:
                    check_file_output(str(updated_output_filepath),
                                      str(existing_output_filepath),
                                      file_format=file_format)
                except AssertionError:
                    pass
                else:
                    include_file = False

            if include_file:
                really_changed_files.append(changed_file)

        # Copy over the new files as well as the really changed files
        new_or_changed_files = new_files + really_changed_files
        for file in new_or_changed_files:
            copyfile(updated_output_path / file, existing_output_path / file)

        # Update the lists with files that were changed for this source
        self.deleted_files.extend([(test_name, file) for file in existing_output_only_files])
        self.updated_files.extend([(test_name, file) for file in new_or_changed_files])


    def update_test_data(self,
                         source,
                         test_tool,
                         skll=False):
        """
        Determine whether we are updating input or output data
        and run ``update_source()`` with the relevant parameters.

        Parameters
        ----------
        source : str
            Name of source directory.
        test_tool : str
            What tool is tested by this test.
        skll : bool, optional
            Whether the given source
            is for a SKLL-based experiment test.
        """
        existing_output_path = self.tests_directory / "data" / "experiments" / source / "output"
        # if we have a tool without with output
        # we update the outputs
        if test_tool in tools_with_output:
            self.update_source(source, skll=skll)
        # if we have a tool with input data we also update inputs
        if test_tool in tools_with_input_data:
            for input_dir in existing_output_path.parent.iterdir():
                if not input_dir.is_dir():
                    continue
                if input_dir.name in ['output', 'figure', 'report']:
                    continue
                else:
                    input_source = input_dir.name
                    self.update_source(source,
                                       skll=skll,
                                       file_type='input',
                                       input_source=input_source)


    def run(self):
        """
        Update all tests found in the files given by the `test_suffixes` class attribute.
        """

        # import all the test_suffix experiment files using SourceFileLoader
        # adapted from: http://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
        for test_suffix in self.test_suffixes:
            test_module_path = join(self.tests_directory, 'test_experiment_{}.py'.format(test_suffix))
            test_module = SourceFileLoader('loaded_{}'.format(test_suffix), test_module_path).load_module()
            test_tool = test_suffix.split('_')[0]

            # skip the module if it tells us that it doesn't want the data for its tests updated
            if hasattr(test_module, '_AUTO_UPDATE'):
                if not test_module._AUTO_UPDATE:
                    continue

            # iterate over all the members and focus on only the experiment functions.
            # For rsmtool/rsmeval we skip over the functions that are decorated with
            # '@raises' since those functions do not need any test data to be updated.
            # For rsmsummarize and rsmcompare we only update the input files for these functions.
            # For the rest, try to get the source since that's what we need to update
            # the test files.
            for member_name, member_object in getmembers(test_module):
                if isfunction(member_object) and member_name.startswith('test_run_experiment'):
                    function = member_object

                    # get the qualified name of the member function
                    member_qualified_name = member_object.__qualname__

                    # check if it has 'raises' in the qualified name
                    # and skip it
                    if 'raises' in member_qualified_name:
                        continue

                    # otherwise first we check if it's the parameterized function and if so
                    # we can easily get the source from the parameter list
                    if member_name.endswith('parameterized'):
                        for param in function.parameterized_input:
                            source_name = param.args[0]
                            skll = param.kwargs.get('skll', False)
                            self.update_test_data(source_name,
                                                  test_tool,
                                                  skll=skll)

                    # if it's another function, then we actually inspect the code
                    # to get the source. Note that this should never be a SKLL experiment
                    # since those should always be run parameterized
                    else:
                        function_code_lines = getsourcelines(function)
                        source_line = [line for line in function_code_lines[0]
                                       if re.search(r'source = ', line)]
                        source_name = eval(source_line[0].strip().split(' = ')[1])
                        self.update_test_data(source_name, test_tool)


    def print_report(self):
        """
        Print a report of all the changes made when the updater was run.
        """
        # print out the number and list of overall deleted files
        print('{} deleted:'.format(len(self.deleted_files)))
        for source, deleted_file in self.deleted_files:
            print('{} {}'.format(source, deleted_file))
        print()

        # find added/updated input files: in this case the source # will consist of
        # the test name and the input test name separated by '/'.
        updated_input_files = [(source, updated_file) for (source, updated_file)
                               in self.updated_files if '/' in source]


        # print out the number and list of overall added/updated non-model files
        print('{} added/updated:'.format(len(self.updated_files)))
        for source, updated_file in self.updated_files:
            print('{} {}'.format(source, updated_file))
        print()

        # now print out missing and/or empty updated output directories
        print('{} missing/empty sources in updated outputs:'.format(len(self.missing_or_empty_sources)))
        for source in self.missing_or_empty_sources:
            print('{}'.format(source))
        print()

        # if we updated any input files, let the user know that they need to
        # re-run the tests and update test outputs
        if len(updated_input_files) > 0:
            print("WARNING: {} input files for rsmcompare/rsmsummarize "
                  "tests have been updated. You need to re-run these "
                  "tests and update test outputs".format(len(updated_input_files)))
