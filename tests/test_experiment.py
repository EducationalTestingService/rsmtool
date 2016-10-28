import warnings

from glob import glob
from os.path import basename, dirname, exists, join

from nose.tools import raises

from rsmtool.test_utils import (check_csv_output,
                                check_report,
                                check_scaled_coefficients,
                                check_subgroup_outputs,
                                check_all_csv_exist,
                                check_consistency_files_exist,
                                do_run_experiment,
                                do_run_evaluation,
                                do_run_prediction,
                                do_run_comparison,
                                do_run_summary)

# get the directory containing the tests
test_dir = dirname(__file__)


def test_run_experiment_lr():
    # basic experiment with a LinearRegression model

    source = 'lr'
    experiment_id = 'lr'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_old_config():
    # basic experiment with a LinearRegression model but using an
    # old style configuration file

    source = 'lr-old-config'
    experiment_id = 'empWt'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))

    # run this experiment but suppress the expected deprecation warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        do_run_experiment(source, experiment_id, config_file)
        output_dir = join('test_outputs', source, 'output')
        expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
        html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

        csv_files = glob(join(output_dir, '*.csv'))
        for csv_file in csv_files:
            csv_filename = basename(csv_file)
            expected_csv_file = join(expected_output_dir, csv_filename)

            if exists(expected_csv_file):
                yield check_csv_output, csv_file, expected_csv_file

        yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
        yield check_scaled_coefficients, source, experiment_id
        yield check_report, html_report


def test_run_experiment_lr_feature_json():
    # basic experiment with a LinearRegression model but using
    # feature json file

    source = 'lr-feature-json'
    experiment_id = 'lr'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))

    # run this experiment but suppress the expected deprecation warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        do_run_experiment(source, experiment_id, config_file)
        output_dir = join('test_outputs', source, 'output')
        expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
        html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

        csv_files = glob(join(output_dir, '*.csv'))
        for csv_file in csv_files:
            csv_filename = basename(csv_file)
            expected_csv_file = join(expected_output_dir, csv_filename)

            if exists(expected_csv_file):
                yield check_csv_output, csv_file, expected_csv_file

        yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
        yield check_scaled_coefficients, source, experiment_id
        yield check_report, html_report


def test_run_experiment_lr_subset_features():
    # basic experiment with LinearRegression model but using only
    # a subset of all the features in the train/test files

    source = 'lr-subset-features'
    experiment_id = 'lr_subset'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_subset_feature_file():
    # basic experiment with LinearRegression model but using only
    # a subset of all the features in the train/test files.
    # The subset is defined usig a feature subset specs file

    source = 'lr-with-feature-subset-file'
    experiment_id = 'lr_with_feature_subset_file'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


@raises(ValueError)
def test_run_experiment_lr_subset_feature_file_and_feature_file():
    # basic experiment with LinearRegression model and a feature file and
    # also a subset file. This is not allowed and so should raise a ValueError.

    source = 'lr-with-feature-subset-file-and-feature-file'
    experiment_id = 'lr_with_feature_subset_file'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)


def test_run_experiment_ridge():

    # basic experiment with ridge model

    source = 'ridge'
    experiment_id = 'Ridge'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'skll'
    yield check_report, html_report

def test_run_experiment_linearsvr():

    # basic experiment with LinearSVR model

    source = 'linearsvr'
    experiment_id = 'LinearSVR'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'skll'
    yield check_report, html_report


def test_run_experiment_lr_subgroups():

    # basic experiment with LinearRegression model but also including
    # subgroup analyses

    source = 'lr-subgroups'
    experiment_id = 'lr_subgroups'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_subgroup_outputs, output_dir, experiment_id, ['L1']
    yield check_report, html_report


def test_run_experiment_lr_numeric_subgroup():

    # basic experiment with LinearRegression model but also including
    # subgroup analyses where one of the subgroups has numeric values

    source = 'lr-with-numeric-subgroup'
    experiment_id = 'lr_subgroups'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_subgroup_outputs, output_dir, experiment_id, ['ITEM', 'QUESTION']
    yield check_report, html_report


def test_run_experiment_lr_eval():

    # basic evaluation experiment using rsmeval

    source = 'lr-eval'
    experiment_id = 'lr_evaluation'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_report, html_report


def test_run_experiment_lr_eval_with_scaling():

    # rsmeval evaluation experiment with scaling

    source = 'lr-eval-with-scaling'
    experiment_id = 'lr_evaluation_with_scaling'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_report, html_report


def test_run_experiment_lr_subgroups_with_edge_cases():

    # this test is identical to lr but with the addition of groups
    # with edge cases such as 1-3 responses per group or a group with
    # the same human score for each response.

    source = 'lr-subgroups-with-edge-cases'
    experiment_id = 'lr_subgroups_with_edge_cases'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))

    # run this experiment but suppress the expected runtime warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        do_run_experiment(source, experiment_id, config_file)
        output_dir = join('test_outputs', source, 'output')
        expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
        html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

        csv_files = glob(join(output_dir, '*.csv'))
        for csv_file in csv_files:
            csv_filename = basename(csv_file)
            expected_csv_file = join(expected_output_dir, csv_filename)

            if exists(expected_csv_file):
                yield check_csv_output, csv_file, expected_csv_file

        yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
        yield check_scaled_coefficients, source, experiment_id
        yield check_subgroup_outputs, output_dir, experiment_id, ['group_edge_cases']
        yield check_report, html_report


def test_run_experiment_lr_predict():

    # basic experiment using rsmpredict

    source = 'lr-predict'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    for csv_file in ['predictions.csv', 'preprocessed_features.csv']:
        output_file = join(output_dir, csv_file)
        expected_output_file = join(expected_output_dir, csv_file)

        yield check_csv_output, output_file, expected_output_file

def test_run_experiment_lr_predict_with_score():

    # rsmpredict experiment with human score

    source = 'lr-predict-with-score'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    for csv_file in ['predictions.csv', 'preprocessed_features.csv']:
        output_file = join(output_dir, csv_file)
        expected_output_file = join(expected_output_dir, csv_file)

        yield check_csv_output, output_file, expected_output_file


def test_run_experiment_lr_predict_missing_values():

    # basic experiment using rsmpredict when the supplied feature file
    # contains reponses with non-numeric feature values

    source = 'lr-predict-missing-values'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    for csv_file in ['predictions.csv', 'predictions_excluded_responses.csv',
                     'preprocessed_features.csv']:
        output_file = join(output_dir, csv_file)
        expected_output_file = join(expected_output_dir, csv_file)

        yield check_csv_output, output_file, expected_output_file



def test_run_experiment_lr_predict_with_subgroups():

    # basic experiment using rsmpredict with subgroups and other columns

    source = 'lr-predict-with-subgroups'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    for csv_file in ['predictions.csv', 'preprocessed_features.csv']:
        output_file = join(output_dir, csv_file)
        expected_output_file = join(expected_output_dir, csv_file)

        yield check_csv_output, output_file, expected_output_file


def test_run_experiment_lr_predict_with_candidate():

    # basic experiment using rsmpredict with candidate column

    source = 'lr-predict-with-candidate'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    for csv_file in ['predictions.csv', 'preprocessed_features.csv']:
        output_file = join(output_dir, csv_file)
        expected_output_file = join(expected_output_dir, csv_file)

        yield check_csv_output, output_file, expected_output_file


def test_run_experiment_lr_predict_illegal_transformations():

    # rsmpredict experiment where the transformations applied to
    # the new data lead to inf or NaN values. This responses should
    # be treated as if the feature values are missing.

    source = 'lr-predict-illegal-transformations'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    for csv_file in ['predictions.csv', 'predictions_excluded_responses.csv',
                     'preprocessed_features.csv']:
        output_file = join(output_dir, csv_file)
        expected_output_file = join(expected_output_dir, csv_file)

        yield check_csv_output, output_file, expected_output_file


def test_run_experiment_lr_rsmtool_and_rsmpredict():

    # this test is to make sure that both rsmtool
    # and rsmpredict generate the same files

    source = 'lr-rsmtool-rsmpredict'
    experiment_id = 'lr_rsmtool_rsmpredict'
    rsmtool_config_file = join(test_dir,
                               'data',
                               'experiments',
                               source,
                               '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, rsmtool_config_file)
    rsmpredict_config_file = join(test_dir,
                                  'data',
                                  'experiments',
                                  source,
                                  'rsmpredict.json')
    do_run_prediction(source, rsmpredict_config_file)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    csv_files = glob(join(output_dir, '*.csv'))
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    # Check the results for  rsmtool
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_scaled_coefficients, source, experiment_id
    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_report, html_report

    # check that the rsmpredict generated the same results
    for csv_pair in [('predictions.csv',
                      '{}_pred_processed.csv'.format(experiment_id)),
                     ('preprocessed_features.csv',
                      '{}_test_preprocessed_features.csv'.format(experiment_id))]:
        output_file = join(output_dir, csv_pair[0])
        expected_output_file = join(expected_output_dir, csv_pair[1])

        yield check_csv_output, output_file, expected_output_file


def test_run_experiment_lr_missing_values():

    # basic experiment using rsmtool when the supplied feature file
    # contains reponses with non-numeric feature values and/or human
    # scores

    source = 'lr-missing-values'
    experiment_id = 'lr_missing_values'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_include_zeros():

    # This test is the same as lr but includes zeros for
    # training and testing. The test data only includes
    # the excluded responses and data composition files.

    source = 'lr-include-zeros'
    experiment_id = 'lr_include_zeros'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report

def test_run_experiment_lr_with_length():

    # basic experiment including length analyses

    source = 'lr-with-length'
    experiment_id = 'lr_with_length'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_subgroups_with_length():

    # basic experiment with length and subgroup analyses

    source = 'lr-subgroups-with-length'
    experiment_id = 'lr_subgroups_with_length'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_with_large_integer_value():
    # basic experiment with LinearRegression model with some
    # values for FEATURE7 (the integer feature) being
    # outside +/-4 SD.

    source = 'lr-with-large-integer-value'
    experiment_id = 'lr_with_large_integer_value'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_missing_length_values():

    # basic experiment with length column that has
    # missing values.

    source = 'lr-with-missing-length-values'
    experiment_id = 'lr_with_missing_length_values'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_length_zero_sd():

    # basic experiment with length column
    # that has 0 standard deviation

    source = 'lr-with-length-zero-sd'
    experiment_id = 'lr_with_length_zero_sd'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_with_h2():

    # basic experiment with second rater analyses

    source = 'lr-with-h2'
    experiment_id = 'lr_with_h2'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_subgroups_with_h2():

    # basic experiment with subgroups and second
    # rater analyses

    source = 'lr-subgroups-with-h2'
    experiment_id = 'lr_subgroups_with_h2'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_consistency_files_exist, csv_files, experiment_id
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_with_h2_include_zeros():

    # basic experiment with second rater analyses
    # and keeping the zeros instead of excluding them

    source = 'lr-with-h2-include-zeros'
    experiment_id = 'lr_with_h2_include_zeros'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_consistency_files_exist, csv_files, experiment_id
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_with_h2_and_length():

    # rsmtool experiment with length and second rater analyses

    source = 'lr-with-h2-and-length'
    experiment_id = 'lr_with_h2_and_length'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_eval_with_h2():

    # basic rsmeval experiment with second rater analyses

    source = 'lr-eval-with-h2'
    experiment_id = 'lr_eval_with_h2'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_report, html_report


def test_run_experiment_lr_eval_with_scaling_and_h2_keep_zeros():

    # basic rsmeval experiment with scaling and second
    # rater analyses

    source = 'lr-eval-with-scaling-and-h2-keep-zeros'
    experiment_id = 'lr_eval_with_scaling_and_h2_keep_zeros'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_consistency_files_exist, csv_files, experiment_id
    yield check_report, html_report

def test_run_experiment_lr_with_h2_named_sc1():

    # basic experiment with second rater analyses
    # but with the test score column called sc2
    # and the test second rater column called sc1

    source = 'lr-with-h2-named-sc1'
    experiment_id = 'lr_with_h2_named_sc1'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_consistency_files_exist, csv_files, experiment_id
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_with_defaults_as_extra_columns():

    # basic experiment that has extra columns that are
    # named the default values for the config options
    # but the options have some other values

    source = 'lr-with-defaults-as-extra-columns'
    experiment_id = 'lr_with_defaults_as_extra_columns'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_consistency_files_exist, csv_files, experiment_id
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_exclude_listwise():

    # basic rsmtool experiment with listwise exclusion of candidates

    source = 'lr-exclude-listwise'
    experiment_id = 'lr_exclude_listwise'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_report, html_report


def test_run_experiment_lr_eval_exclude_listwise():

    # basic rsmeval experiment with listwise exclusion of candidates

    source = 'lr-eval-exclude-listwise'
    experiment_id = 'lr_eval_exclude_listwise'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_report, html_report




def test_run_experiment_lr_eval_with_missing_scores():

    # basic rsmeval experiment with missing human scores

    source = 'lr-eval-with-missing-scores'
    experiment_id = 'lr_eval_with_missing_scores'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_report, html_report


def test_run_experiment_lr_eval_with_h2_named_sc1():

    # basic rsmeval experiment with second rater analyses
    # but the label for the second rater is sc1 and there are
    # missing values for the first score

    source = 'lr-eval-with-h2-named-sc1'
    experiment_id = 'lr_eval_with_h2_named_sc1'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_consistency_files_exist, csv_files, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_eval_with_missing_data():

    # basic rsmeval experiment with missing machine and human scores

    source = 'lr-eval-with-missing-data'
    experiment_id = 'lr_eval_with_missing_data'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_report, html_report


def test_run_experiment_lr_with_custom_order():

    # rsmtool experiment with custom section ordering

    source = 'lr-with-custom-order'
    experiment_id = 'lr_with_custom_order'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_eval_with_custom_order():

    # rsmeval experiment with custom section ordering

    source = 'lr-eval-with-custom-order'
    experiment_id = 'lr_eval_with_custom_order'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_report, html_report


def test_run_experiment_lr_with_custom_sections():

    # rsmtool experiment with custom sections

    source = 'lr-with-custom-sections'
    experiment_id = 'lr_with_custom_sections'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_eval_with_custom_sections():

    # rsmeval experiment with custom sections

    source = 'lr-eval-with-custom-sections'
    experiment_id = 'lr_eval_with_custom_sections'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_report, html_report


def test_run_experiment_lr_with_custom_sections_and_order():

    # rsmtool experiment with custom sections and custom
    # section ordering

    source = 'lr-with-custom-sections-and-order'
    experiment_id = 'lr_with_custom_sections_and_order'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report




def test_run_experiment_lr_eval_with_custom_sections_and_order():

    # rsmeval experiment with custom sections and custom section
    # ordering

    source = 'lr-eval-with-custom-sections-and-order'
    experiment_id = 'lr_eval_with_custom_sections_and_order'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_report, html_report


def test_run_experiment_lr_exclude_flags():

    # rsmtool experiment with LinearRegression model with filtering
    # responses based on a flag column

    source = 'lr-exclude-flags'
    experiment_id = 'lr_exclude_flags'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_exclude_flags_and_zeros():

    # rsmtool experiment with LinearRegression model with filtering
    # responses based on a flag column and also filtering zero scores
    # in the training data

    source = 'lr-exclude-flags-and-zeros'
    experiment_id = 'lr_exclude_flags_and_zeros'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_eval_exclude_flags():

    # evaluation experiment using rsmeval but with excluded responses
    # using flag columns

    source = 'lr-eval-exclude-flags'
    experiment_id = 'lr_eval_exclude_flags'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_report, html_report


def test_run_experiment_lr_use_all_features():

    # rsmtool experiment with no feature file specified:
    # the tool should be using all columns not assigned to
    # anything else.

    source = 'lr-use-all-features'
    experiment_id = 'lr_use_all_features'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_equalweightslr_model():
    # basic rsmtool experiment test using the EqualWeightsLR model

    source = 'equalweightslr'
    experiment_id = 'equalweightslr'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_rebalancedlr_model():
    # basic rsmtool experiment test using the RebalancedLR model

    source = 'rebalancedlr'
    experiment_id = 'rebalancedlr'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lassofixedlambdathenlr_model():
    # basic rsmtool experiment test using the LassoFixedLambdaThenLR model

    source = 'lassofixedlambdathenlr'
    experiment_id = 'lassofixedlambdathenlr'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_positivelassocvthenlr_model():
    # basic rsmtool experiment test using the PositiveLassoCVThenLR model

    source = 'positivelassocvthenlr'
    experiment_id = 'positivelassocvthenlr'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_nnlr_model():
    # basic rsmtool experiment test using the NNLR model

    source = 'nnlr'
    experiment_id = 'nnlr'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lassofixedlambdathennnlr_model():
    # basic rsmtool experiment test using the LassoFixedLambdaThenNNLR model

    source = 'lassofixedlambdathennnlr'
    experiment_id = 'lassofixedlambdathennnlr'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lassofixedlambda_model():
    # basic rsmtool experiment test using the LassoFixedLambda model

    source = 'lassofixedlambda'
    experiment_id = 'lassofixedlambda'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_positivelassocv_model():
    # basic rsmtool experiment test using the PositiveLassoCV model

    source = 'positivelassocv'
    experiment_id = 'positivelassocv'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_candidate_same_as_id():
    # basic LR experiment but with `candidate_column` set to the
    # same value as `id_column`

    source = 'lr-candidate-same-as-id'
    experiment_id = 'lr_candidate_same_as_id'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_candidate_same_as_id_candidate():
    # basic LR experiment but with `candidate_column` set to the
    # same value as `id_column` and both values are set to 'candidate'

    source = 'lr-candidate-same-as-id-candidate'
    experiment_id = 'lr_candidate_same_as_id_candidate'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_compare():

    # basic rsmcompare experiment comparing a LinearRegression
    # experiment to itself
    source = 'lr-self-compare'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmcompare.json')
    do_run_comparison(source, config_file)

    html_report = join('test_outputs', source, 'lr_subgroups_vs_lr_subgroups_report.html')
    yield check_report, html_report


def test_run_experiment_lr_compare_different_experiments():

    # basic rsmcompare experiment comparing two Linear regression
    # experiments with different features and small differences in
    # training and evaluation sets
    source = 'lr-different-compare'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmcompare.json')
    do_run_comparison(source, config_file)

    html_report = join('test_outputs', source, 'lr_baseline_vs_lr_with_FEATURE8_and_zero_scores_report.html')
    yield check_report, html_report

def test_run_experiment_lr_compare_with_h2():

    # basic rsmcompare experiment comparing a LinearRegression
    # experiment to itself where the rsmtool report contains
    # h2 information
    source = 'lr-self-compare-with-h2'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmcompare.json')
    do_run_comparison(source, config_file)

    html_report = join('test_outputs', source, 'lr_with_h2_vs_lr_with_h2_report.html')
    yield check_report, html_report


def test_run_experiment_lr_compare_with_custom_order():

    # basic rsmcompare experiment comparing a LinearRegression
    # experiment to itself with a custom list of sections
    source = 'lr-self-compare-with-custom-order'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmcompare.json')
    do_run_comparison(source, config_file)

    html_report = join('test_outputs', source, 'lr_subgroups_vs_lr_subgroups_report.html')
    yield check_report, html_report


def test_run_experiment_lr_compare_with_chosen_sections():

    # basic rsmcompare experiment comparing a LinearRegression
    # experiment to itself with a custom list of sections
    source = 'lr-self-compare-with-chosen-sections'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmcompare.json')
    do_run_comparison(source, config_file)

    html_report = join('test_outputs', source, 'lr_subgroups_vs_lr_subgroups_report.html')
    yield check_report, html_report


def test_run_experiment_lr_compare_with_custom_sections_and_custom_order():

    # basic rsmcompare experiment comparing a LinearRegression
    # experiment to itself with custom sections included and
    # all sections in a custom order
    source = 'lr-self-compare-with-custom-sections-and-custom-order'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmcompare.json')
    do_run_comparison(source, config_file)

    html_report = join('test_outputs', source, 'lr_subgroups_vs_lr_subgroups_report.html')
    yield check_report, html_report


def test_run_experiment_linearsvr_compare():

    # basic rsmcompare experiment comparing an experiment
    # which uses a SKLL model to itself
    source = 'linearsvr-self-compare'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmcompare.json')
    do_run_comparison(source, config_file)

    html_report = join('test_outputs', source, 'LinearSVR_vs_LinearSVR_report.html')
    yield check_report, html_report


def test_run_experiment_lr_eval_compare():

    # basic rsmcompare experiment comparing an rsmeval
    # experiment to itself
    source = 'lr-eval-self-compare'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmcompare.json')
    do_run_comparison(source, config_file)

    html_report = join('test_outputs', source, 'lr_eval_with_h2_vs_lr_eval_with_h2_report.html')
    yield check_report, html_report


def test_run_experiment_lr_eval_tool_compare():

    # basic rsmcompare experiment comparing an rsmeval
    # experiment to an rsmtool experiment
    source = 'lr-eval-tool-compare'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmcompare.json')
    do_run_comparison(source, config_file)

    html_report = join('test_outputs', source, 'lr_with_h2_vs_lr_eval_with_h2_report.html')
    yield check_report, html_report


def test_run_experiment_lr_summary():

    # basic rsmsummarize experiment comparing several rsmtool experiments
    source = 'lr-self-summary'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmsummarize.json')
    do_run_summary(source, config_file)

    html_report = join('test_outputs', source, 'model_comparison_report.html')
    yield check_report, html_report


def test_run_experiment_linearsvr_summary():

    # basic rsmsummarize experiment comparing an experiment
    # which uses a SKLL model to itself
    source = 'linearsvr-self-summary'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmsummarize.json')
    do_run_summary(source, config_file)

    html_report = join('test_outputs', source, 'model_comparison_report.html')
    yield check_report, html_report


def test_run_experiment_lr_eval_summary():

    # basic rsmsummarize experiment comparing an rsmtool and rsmeval experiments
    source = 'lr-self-eval-summary'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmsummarize.json')
    do_run_summary(source, config_file)

    html_report = join('test_outputs', source, 'model_comparison_report.html')
    yield check_report, html_report


def test_run_experiment_lr_summary_with_custom_sections_and_custom_order():

    # basic rsmsummarize experiment comparing a LinearRegression
    # experiment to itself with a custom list of sections
    source = 'lr-self-summary-with-custom-sections'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmsummarize.json')
    do_run_summary(source, config_file)

    html_report = join('test_outputs', source, 'model_comparison_report.html')
    yield check_report, html_report

def test_run_experiment_lr_tsv_input_files():

    # rsmtool experiment with input files in .tsv format

    source = 'lr-tsv-input-files'
    experiment_id = 'lr_tsv_input_files'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_tsv_input_and_subset_files():

    # rsmtool experiment with input files in .tsv format
    # including a feature subset file in .tsv format

    source = 'lr-tsv-input-and-subset-files'
    experiment_id = 'lr_tsv_input_and_subset_files'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_xlsx_input_files():

    # rsmtool experiment with input files in .xlsx format

    source = 'lr-xlsx-input-files'
    experiment_id = 'lr_xlsx_input_files'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_xlsx_input_and_subset_files():

    # rsmtool experiment with input files in .xlsx format
    # including a feature subset file in .xlsx format

    source = 'lr-xlsx-input-and-subset-files'
    experiment_id = 'lr_xlsx_input_and_subset_files'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_all_csv_exist, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_eval_tsv_input_files():

    # rsmeval experiment input file as in .tsv format

    source = 'lr-eval-tsv-input-files'
    experiment_id = 'lr_evaluation_with_scaling'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_report, html_report


def test_run_experiment_lr_eval_xlsx_input_files():

    # rsmeval experiment input file as in .xlsx format

    source = 'lr-eval-xlsx-input-files'
    experiment_id = 'lr_evaluation_with_scaling'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_csv_output, csv_file, expected_csv_file

    yield check_report, html_report


def test_run_experiment_lr_predict_tsv_input_files():

    # rsmpredict experiment with input file in .tsv format

    source = 'lr-predict-tsv-input-files'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    for csv_file in ['predictions.csv', 'preprocessed_features.csv']:
        output_file = join(output_dir, csv_file)
        expected_output_file = join(expected_output_dir, csv_file)

        yield check_csv_output, output_file, expected_output_file


def test_run_experiment_lr_predict_xlsx_input_files():

    # rsmpredict experiment with input file in .xlsx format

    source = 'lr-predict-xlsx-input-files'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    for csv_file in ['predictions.csv', 'preprocessed_features.csv']:
        output_file = join(output_dir, csv_file)
        expected_output_file = join(expected_output_dir, csv_file)

        yield check_csv_output, output_file, expected_output_file


@raises(ValueError)
def test_run_experiment_lr_length_column_and_feature():

    # rsmtool experiment that has length column but
    # the same column as a model feature
    source = 'lr-with-length-and-feature'
    experiment_id = 'lr_with_length_and_feature'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_h2_column_and_feature():

    # rsmtool experiment that has second rater column but
    # the same column as a model feature
    source = 'lr-with-h2-and-feature'
    experiment_id = 'lr_with_h2_and_feature'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_same_h1_and_h2():

    # rsmtool experiment that has label column
    # and second rater column set the same

    source = 'lr-same-h1-and-h2'
    experiment_id = 'lr_same_h1_and_h2'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_with_repeated_ids():

    # rsmtool experiment with non-unique ids
    source = 'lr-with-repeated-ids'
    experiment_id = 'lr_with_repeated_ids'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_eval_with_repeated_ids():

    # rsmeval experiment with non-unique ids
    source = 'lr-eval-with-repeated-ids'
    experiment_id = 'lr_eval_with_repeated_ids'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_predict_with_repeated_ids():

    # rsmpredict experiment with non-unique ids
    source = 'lr-predict-with-repeated-ids'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(ValueError)
def test_run_experiment_lr_with_sc1_as_feature_name():

    # rsmtool experiment with sc1 used as the name of a feature
    source = 'lr-with-sc1-as-feature-name'
    experiment_id = 'lr_with_sc1_as_feature_name'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_with_length_as_feature_name():

    # rsmtool experiment with 'length' used as feature name
    # when a length analysis is requested using a different feature
    source = 'lr-with-length-as-feature-name'
    experiment_id = 'lr_with_length_as_feature_name'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_with_sc2_as_feature_name():

    # rsmtool experiment with sc2 used as a feature name
    # when the user also requests h2 analysis using a different
    # column
    source = 'lr-with-sc2-as-feature-name'
    experiment_id = 'lr_with_sc2_as_feature_name'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_with_subgroup_as_feature_name():

    # rsmtool experiment with a subgroup name used as feature
    # when the user also requests subgroup analysis with that subgroup

    source = 'lr-with-subgroup-as-feature-name'
    experiment_id = 'lr_with_subgroup_as_feature_name'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_all_non_numeric_scores():

    # rsmtool experiment with all values for `sc1`
    # being non-numeric and all getting filtered out
    # which should raise an exception

    source = 'lr-with-all-non-numeric-scores'
    experiment_id = 'lr_with_all_non_numeric_scores'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_eval_all_non_numeric_scores():

    # rsmeval experiment with all values for the human
    # score being non-numeric and all getting filtered out
    # which should raise an exception

    source = 'lr-eval-with-all-non-numeric-scores'
    experiment_id = 'lr_eval_all_non_numeric_scores'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)

@raises(ValueError)
def test_run_experiment_lr_eval_same_system_human_score():

    # rsmeval experiment with the same value supplied
    # for both human score ans system score

    source = 'lr-eval-same-system-human-score'
    experiment_id = 'lr_eval_same_system_human_score'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_eval_all_non_numeric_machine_scores():

    # rsmeval experiment with all the machine scores`
    # being non-numeric and all getting filtered out
    # which should raise an exception

    source = 'lr-eval-with-all-non-numeric-machine-scores'
    experiment_id = 'lr_eval_all_non_numeric_machine_scores'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)


@raises(KeyError)
def test_run_experiment_eval_lr_with_missing_h2_column():

    # rsmeval experiment with `second_human_score_column`
    # set to a column that does not exist in the given
    # predictions file
    source = 'lr-eval-with-missing-h2-column'
    experiment_id = 'lr_eval_with_missing_h2_column'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)


@raises(KeyError)
def test_run_experiment_eval_lr_with_missing_candidate_column():

    # rsmeval experiment with `candidate_column`
    # set to a column that does not exist in the given
    # predictions file
    source = 'lr-eval-with-missing-candidate-column'
    experiment_id = 'lr_eval_with_missing_candidate_column'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_one_fully_non_numeric_feature():

    # rsmtool experiment with all values for one of the
    # features being non-numeric and all getting filtered out
    # which should raise an exception

    source = 'lr-with-one-fully-non-numeric-feature'
    experiment_id = 'lr_with_one_fully_non_numeric_feature'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_none_flagged():

    # rsmtool experiment where all responses have the bad flag
    # value and so they all get filtered out which should
    # raise an exception

    source = 'lr-with-none-flagged'
    experiment_id = 'lr_with_none_flagged'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)


@raises(FileNotFoundError)
def test_run_experiment_lr_predict_missing_model_file():

    # rsmpredict experiment with missing model file
    source = 'lr-predict-missing-model-file'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_lr_predict_missing_feature_file():

    # rsmpredict experiment with missing feature file
    source = 'lr-predict-missing-feature-file'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_lr_predict_missing_postprocessing_file():

    # rsmpredict experiment with missing post-processing file
    source = 'lr-predict-missing-postprocessing-file'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_predict_no_input_feature_file():

    # rsmpredict experiment with missing feature file
    source = 'lr-predict-no-input-feature-file'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_predict_no_experiment_dir():

    # rsmpredict experiment with missing experiment dir
    source = 'lr-predict-no-experiment-dir'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_predict_no_output_dir():

    # rsmpredict experiment where experiment_dir
    # does not containt output directory
    source = 'lr-predict-no-output-dir'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_predict_no_experiment_id():

    # rsmpredict experiment ehere the experiment_dir
    # does not contain the experiment with the stated id
    source = 'lr-predict-no-experiment-id'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(KeyError)
def test_run_experiment_lr_predict_missing_columns():

    # rsmpredict experiment with missing columns
    # from the config file
    source = 'lr-predict-missing-columns'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)

@raises(KeyError)
def test_run_experiment_lr_predict_missing_feature():

    # rsmpredict experiment with missing features
    source = 'lr-predict-missing-feature'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(ValueError)
def test_run_experiment_lr_predict_no_numeric_feature_values():

    # rsmpredict experiment with missing post-processing file
    source = 'lr-predict-no-numeric-feature-values'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(ValueError)
def test_run_experiment_wrong_model_name():

    # rsmtool experiment with incorrect model name
    source = 'wrong-model-name'
    experiment_id = 'wrong_model_name'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_empwtdropneg():

    # rsmtool experiment with no longer supported empWtDropNeg model
    source = 'empwtdropneg'
    experiment_id = 'empWtDropNeg'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_duplicate_feature_names():

    # rsmtool experiment with duplicate feature names
    source = 'lr-with-duplicate-feature-names'
    experiment_id = 'lr_with_duplicate_feature_names'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)

@raises(FileNotFoundError)
def test_run_experiment_lr_compare_wrong_directory():

    # basic rsmcompare experiment comparing a LinearRegression
    # experiment to itself
    source = 'lr-self-compare-wrong-directory'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmcompare.json')
    do_run_comparison(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_lr_compare_wrong_experiment_id():

    # basic rsmcompare experiment comparing a LinearRegression
    # experiment to itself
    source = 'lr-self-compare-wrong-id'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmcompare.json')
    do_run_comparison(source, config_file)

@raises(FileNotFoundError)
def test_run_experiment_summary_wrong_directory():

    # rsmsummarize experiment where the specified directory
    # does not exist
    source = 'summary-wrong-directory'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmsummarize.json')
    do_run_summary(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_summary_no_csv_directory():

    # rsmsummarize experiment where the specified directory
    # does not contain any rsmtool experiments
    source = 'summary-no-output-dir'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmsummarize.json')
    do_run_summary(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_summary_no_json():

    # rsmsummarize experiment where the specified directory
    # does not contain any json files
    source = 'summary-no-json-file'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmsummarize.json')
    do_run_summary(source, config_file)


