import re
import warnings

import numpy as np

from glob import glob
from os import remove
from os.path import basename, exists, join
from pathlib import Path

from nose.tools import assert_equal, ok_
from pandas.util.testing import assert_frame_equal

from rsmtool.reader import DataReader
from rsmtool.modeler import Modeler
from rsmtool.rsmtool import run_experiment
from rsmtool.rsmcompare import run_comparison
from rsmtool.rsmeval import run_evaluation
from rsmtool.rsmpredict import compute_and_save_predictions
from rsmtool.rsmsummarize import run_summary

html_error_regexp = re.compile(r'Traceback \(most recent call last\)')
html_warning_regexp = re.compile(r'<div class=".*?output_stderr.*?>')
section_regexp = re.compile(r'<h2>(.*?)</h2>')

# get the directory containing the tests
rsmtool_test_dir = Path(__file__).absolute().parent.parent.joinpath('tests')


def check_run_experiment(source,
                         experiment_id,
                         subgroups=None,
                         consistency=False,
                         skll=False,
                         file_format='csv',
                         given_test_dir=None):
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
    """
    # use the test directory from this file unless it's been overridden
    test_dir = given_test_dir if given_test_dir else rsmtool_test_dir

    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))

    model_type = 'skll' if skll else 'rsmtool'

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        do_run_experiment(source, experiment_id, config_file)

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

    check_report(html_report)


def check_run_evaluation(source,
                         experiment_id,
                         subgroups=None,
                         consistency=False,
                         file_format='csv',
                         given_test_dir=None):
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
    """
    # use the test directory from this file unless it's been overridden
    test_dir = given_test_dir if given_test_dir else rsmtool_test_dir

    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        do_run_evaluation(source, experiment_id, config_file)

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

    check_report(html_report)


def check_run_comparison(source, experiment_id, given_test_dir=None):
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
    """
    # use the test directory from this file unless it's been overridden
    test_dir = given_test_dir if given_test_dir else rsmtool_test_dir

    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmcompare.json')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        do_run_comparison(source, config_file)

    html_report = join('test_outputs', source, '{}.html'.format(experiment_id))
    check_report(html_report)


def check_run_prediction(source, excluded=False, file_format='csv', given_test_dir=None):
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
    """
    # use the test directory from this file unless it's been overridden
    test_dir = given_test_dir if given_test_dir else rsmtool_test_dir

    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        do_run_prediction(source, config_file)

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


def check_run_summary(source, file_format='csv', given_test_dir=None):
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
    """
    # use the test directory from this file unless it's been overridden
    test_dir = given_test_dir if given_test_dir else rsmtool_test_dir

    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmsummarize.json')
    do_run_summary(source, config_file)

    html_report = join('test_outputs', source, 'report', 'model_comparison_report.html')

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    output_files = glob(join(output_dir, '*.{}'.format(file_format)))
    for output_file in output_files:
        output_filename = basename(output_file)
        expected_output_file = join(expected_output_dir, output_filename)

        if exists(expected_output_file):
            check_file_output(output_file, expected_output_file)

    check_report(html_report)


def do_run_experiment(source, experiment_id, config_file):
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
    config_file : str
        Path to the experiment configuration file.
    """
    source_output_dir = 'test_outputs'
    experiment_dir = join(source_output_dir, source)

    # remove all previously created files
    for output_subdir in ['output', 'figure', 'report']:
        files = glob(join(source_output_dir, source, output_subdir, '*'))
        for f in files:
            remove(f)
    run_experiment(config_file, experiment_dir)


def do_run_evaluation(source, experiment_id, config_file):
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
    config_file : str
        Path to the experiment configuration file.
    """
    source_output_dir = 'test_outputs'
    experiment_dir = join(source_output_dir, source)

    # remove all previously created files
    for output_subdir in ['output', 'figure', 'report']:
        files = glob(join(source_output_dir, source, output_subdir, '*'))
        for f in files:
            remove(f)
    run_evaluation(config_file, experiment_dir)


def do_run_prediction(source, config_file):
    """
    Run RSMPredict experiment using the given experiment
    configuration file located in the given source directory.

    Parameters
    ----------
    source : str
        Path to where the test is located on disk.
    config_file : str
        Path to the experiment configuration file.
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

    compute_and_save_predictions(config_file, output_file, feats_file)


def do_run_comparison(source, config_file):
    """
    Run RSMCompare experiment using the given experiment
    configuration file located in the given source directory.

    Parameters
    ----------
    source : str
        Path to where the test is located on disk.
    config_file : str
        Path to the experiment configuration file.
    """
    source_output_dir = 'test_outputs'
    experiment_dir = join(source_output_dir, source)
    run_comparison(config_file, experiment_dir)


def do_run_summary(source, config_file):
    """
    Run rsmsummarizeary experiment using the given experiment
    configuration file located in the given source directory.

    Parameters
    ----------
    source : str
        Path to where the test is located on disk.
    config_file : str
        Path to the experiment configuration file.
    """
    source_output_dir = 'test_outputs'
    experiment_dir = join(source_output_dir, source)

    # remove all previously created files
    for output_subdir in ['output', 'figure', 'report']:
        files = glob(join(source_output_dir, source, output_subdir, '*'))
        for f in files:
            remove(f)
    run_summary(config_file, experiment_dir)


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

    # if the first column is numeric, just force the index to string;
    # however, if it is non-numeric, set it as the index and then
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

    # convert any integer columns to floats in either data frame
    for df in [df1, df2]:
        for c in df.columns:
            if df[c].dtype == np.int64:
                df[c] = df[c].astype(np.float64)

    # do the same for indices
    for df in [df1, df2]:
        if df.index.dtype == np.int64:
            df.index = df.index.astype(np.float64)

    # for pca and factor correlations convert all values to absolutes
    # because the sign may not always be the same
    if (file1.endswith('pca.{}'.format(file_format)) or
        file1.endswith('factor_correlations.{}'.format(file_format))):
        for df in [df1, df2]:
            msk = df.dtypes == np.float64
            df.loc[:, msk] = df.loc[:, msk].abs()

    try:
        assert_frame_equal(df1.sort_index(axis=1),
                           df2.sort_index(axis=1),
                           check_exact=False,
                           check_less_precise=True)
    except AssertionError as e:
        message = e.args[0]
        new_message = 'File {} - {}'.format(basename(file1), message)
        e.args = (new_message, )
        raise


def check_report(html_file):
    """
    Checks if the HTML report contains any errors.
    Raises an AssertionError if it does.

    Parameters
    ----------
    html_file : str
        Path the HTML report file on disk.
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
    assert_equal(report_errors, 0)
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
                       check_less_precise=True)


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
                      "_degradation.{}".format(file_format)]

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
