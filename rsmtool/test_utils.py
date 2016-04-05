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

html_error_regexp = re.compile(r'Traceback \(most recent call last\)')
section_regexp = re.compile(r'<h2>(.*?)</h2>')

# utility functions to run the experiments
def do_run_experiment(source, experiment_id, config_file):
    source_output_dir = 'test_outputs'
    experiment_dir = join(source_output_dir, source)

    # remove all previously created files
    for output_subdir in ['output', 'figure', 'report']:
        files = glob(join(source_output_dir, source, output_subdir, '*'))
        for f in files:
            remove(f)
    run_experiment(config_file, experiment_dir)


def do_run_evaluation(source, experiment_id, config_file):
    source_output_dir = 'test_outputs'
    experiment_dir = join(source_output_dir, source)

    # remove all previously created files
    for output_subdir in ['output', 'figure', 'report']:
        files = glob(join(source_output_dir, source, output_subdir, '*'))
        for f in files:
            remove(f)
    run_evaluation(config_file, experiment_dir)


def do_run_prediction(source, config_file):
    source_output_dir = 'test_outputs'
    output_file = join(source_output_dir, source, 'output', 'predictions.csv')
    feats_file = join(source_output_dir, source, 'output', 'preprocessed_features.csv')

    # remove all previously created files
    files = glob(join(source_output_dir, 'output', '*'))
    for f in files:
        remove(f)

    compute_and_save_predictions(config_file, output_file, feats_file)


def do_run_comparison(source, config_file):
    source_output_dir = 'test_outputs'
    experiment_dir = join(source_output_dir, source)
    run_comparison(config_file, experiment_dir)

# check if two csv files have values that are the same
# to within three decimal places
def check_csv_output(csv1, csv2):
    df1 = pd.read_csv(csv1, index_col=0)
    df2 = pd.read_csv(csv2, index_col=0)

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
    if csv1.endswith('pca.csv') or csv1.endswith('factor_correlations.csv'):
        for df in [df1, df2]:
            msk = df.dtypes == np.float64
            df.loc[:,msk] = df.loc[:,msk].abs()

    assert_frame_equal(df1.sort_index(axis=1),
                       df2.sort_index(axis=1),
                       check_exact=False,
                       check_less_precise=True)


def check_report(html_file):
    report_errors = 0
    with open(html_file, 'r') as htmlf:
        for line in htmlf:
            m = html_error_regexp.search(line)
            if m:
                report_errors += 1
    assert_equal(report_errors, 0)


def check_scaled_coefficients(source, experiment_id):
    # check that the predictions generated using scaled
    # coefficients match the scaled scores.

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


# check that all crucial output files have been generated.
def check_all_csv_exist(csv_files, experiment_id, model_source):

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

    csv_must_have = ["_consistency.csv",
                     "_degradation.csv"]

    csv_must_with_id = [experiment_id + file_name for file_name in csv_must_have]
    csv_exist = [basename(file_name) for file_name in csv_files]
    missing_csv = set(csv_must_with_id).difference(set(csv_exist))
    assert_equal(len(missing_csv), 0, "Missing csv files: {}".format(','.join(missing_csv)))


def check_subgroup_outputs(output_dir, experiment_id, subgroups):
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
