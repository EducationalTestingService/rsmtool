import re

import numpy as np
import pandas as pd

from glob import glob
from os import remove
from os.path import basename, join

from nose.tools import assert_equal, ok_
from pandas.util.testing import assert_frame_equal

from rsmtool import run_experiment
from rsmtool.model import create_fake_skll_learner
from rsmtool.predict import predict_with_model
from rsmtool.rsmcompare import run_comparison
from rsmtool.rsmeval import run_evaluation
from rsmtool.rsmpredict import compute_and_save_predictions
from rsmtool.rsmsummarize import run_summary

html_error_regexp = re.compile(r'Traceback \(most recent call last\)')
html_warning_regexp = re.compile(r'<div class=".*?output_stderr.*?>')
section_regexp = re.compile(r'<h2>(.*?)</h2>')


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
    run_summary(config_file, experiment_dir)


def check_csv_output(csv1, csv2):
    """
    Check if two experiment CSV files have values that are
    the same to within three decimal places. Raises an
    AssertionError if they are not.

    Parameters
    ----------
    csv1 : str
        Path to the first CSV file.
    csv2 : str
        Path to the second CSV files.

    """

    # make sure that the main id columns are read as strings since
    # this may affect merging in custom notebooks
    string_columns = ['spkitemid', 'candidate']

    converter_dict = dict([(column, str) for column in string_columns if column])

    df1 = pd.read_csv(csv1, converters=converter_dict)
    df2 = pd.read_csv(csv2, converters=converter_dict)

    # set the first column to be the index. We do it this way to ensure
    # string indices are preserved as such
    df1.index = df1[df1.columns[0]]
    df2.index = df2[df2.columns[0]]

    # sort all the indices alphabetically
    df1.sort_index(inplace=True)
    df2.sort_index(inplace=True)

    # convert any integer columns to floats in either data frame
    for df in [df1, df2]:
        for c in df.columns:
            if not c in string_columns and df[c].dtype == np.int64:
                df[c] = df[c].astype(np.float64)

    # do the same for indices
    #for df in [df1, df2]:
    #    if df.index.dtype == np.int64:
    #        df.index = df.index.astype(np.float64)

    # for pca and factor correlations convert all values to absolutes
    # because the sign may not always be the same
    if csv1.endswith('pca.csv') or csv1.endswith('factor_correlations.csv'):
        for df in [df1, df2]:
            msk = df.dtypes == np.float64
            df.loc[:, msk] = df.loc[:, msk].abs()

    assert_frame_equal(df1.sort_index(axis=1),
                       df2.sort_index(axis=1),
                       check_exact=False,
                       check_less_precise=True)


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


def check_scaled_coefficients(source, experiment_id):
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
    """
    preprocessed_test_file = join('test_outputs',
                                  source,
                                  'output',
                                  '{}_test_preprocessed_features.csv'.format(experiment_id))
    scaled_coefficients_file = join('test_outputs',
                                    source,
                                    'output',
                                    '{}_coefficients_scaled.csv'.format(experiment_id))
    predictions_file = join('test_outputs',
                            source,
                            'output',
                            '{}_pred_processed.csv'.format(experiment_id))

    df_preprocessed_test_data = pd.read_csv(preprocessed_test_file)
    df_old_predictions = pd.read_csv(predictions_file)
    df_old_predictions = df_old_predictions[['spkitemid', 'sc1', 'scale']]

    # create fake skll objects with new coefficients
    df_coef = pd.read_csv(scaled_coefficients_file)
    new_model = create_fake_skll_learner(df_coef)

    # generate new predictions and rename the prediction column to 'scale'
    df_new_predictions = predict_with_model(new_model, df_preprocessed_test_data)
    df_new_predictions.rename(columns={'raw': 'scale'}, inplace=True)

    # check that new predictions match the scaled old predictions
    assert_frame_equal(df_new_predictions.sort_index(axis=1),
                       df_old_predictions.sort_index(axis=1),
                       check_exact=False,
                       check_less_precise=True)


def check_all_csv_exist(csv_files, experiment_id, model_source):
    """
    Check that all crucial output files have been generated.
    Raises an AssertionError if they have not.

    Parameters
    ----------
    csv_files : list of str
        List of CSV files generated by a test.
    experiment_id : str
        The experiment ID.
    model_source : str
        'rsmtool' or 'skll'
    """
    csv_must_have_both = ["_confMatrix.csv",
                          "_cors_orig.csv",
                          "_cors_processed.csv",
                          "_eval.csv",
                          "_eval_short.csv",
                          "_feature.csv",
                          "_feature_descriptives.csv",
                          "_feature_descriptivesExtra.csv",
                          "_feature_outliers.csv",
                          "_margcor_score_all_data.csv",
                          "_pca.csv",
                          "_pcavar.csv",
                          "_pcor_score_all_data.csv",
                          #"_pred.csv", check again
                          "_pred_processed.csv",
                          "_pred_train.csv",
                          "_score_dist.csv",
                          "_train_preprocessed_features.csv",
                          "_test_preprocessed_features.csv",
                          "_postprocessing_params.csv"
                          ]

    csv_must_have_rsmtool = ["_betas.csv",
                             "_coefficients.csv"]
    if model_source == 'rsmtool':
        csv_must_have = csv_must_have_both + csv_must_have_rsmtool
    else:
        csv_must_have = csv_must_have_both

    csv_must_with_id = [experiment_id + file_name for file_name in csv_must_have]
    csv_exist = [basename(file_name) for file_name in csv_files]
    missing_csv = set(csv_must_with_id).difference(set(csv_exist))
    assert_equal(len(missing_csv), 0, "Missing csv files: {}".format(','.join(missing_csv)))


def check_consistency_files_exist(csv_files, experiment_id):
    """
    Check to make sure that the consistency files
    were generated. Raises an AssertionError if
    they were not.

    Parameters
    ----------
    csv_files : list of str
        List of CSV files generated by a test.
    experiment_id : str
        The experiment ID.
    """
    csv_must_have = ["_consistency.csv",
                     "_degradation.csv"]

    csv_must_with_id = [experiment_id + file_name for file_name in csv_must_have]
    csv_exist = [basename(file_name) for file_name in csv_files]
    missing_csv = set(csv_must_with_id).difference(set(csv_exist))
    assert_equal(len(missing_csv), 0, "Missing csv files: {}".format(','.join(missing_csv)))


def check_subgroup_outputs(output_dir, experiment_id, subgroups):
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
    """
    train_preprocessed = pd.read_csv(join(output_dir,
                                          '{}_{}'.format(experiment_id,
                                                         'train_metadata.csv')),
                                     index_col=0)
    test_preprocessed = pd.read_csv(join(output_dir,
                                         '{}_{}'.format(experiment_id,
                                                        'test_metadata.csv')),
                                    index_col=0)
    for group in subgroups:
        ok_(group in train_preprocessed.columns)
        ok_(group in test_preprocessed.columns)

    # check that the total sum of N per category matches the total N
    # in data composition and the total N categories matches what is
    # in overall data composition
    df_data_composition_all = pd.read_csv(join(output_dir,
                                               '{}_data_composition.csv'.format(experiment_id)))
    for group in subgroups:
        composition_by_group = pd.read_csv(join(output_dir,
                                                '{}_data_composition_by_{}.csv'.format(experiment_id,
                                                                                       group)))
        for partition in ['Training', 'Evaluation']:
            partition_info = df_data_composition_all.ix[df_data_composition_all['partition'] == partition]
            ok_(sum(composition_by_group['{} set'.format(partition)]) == partition_info.iloc[0]['responses'])
            ok_(len(composition_by_group.ix[composition_by_group['{} set'.format(partition)] != 0]) == partition_info.iloc[0][group])
