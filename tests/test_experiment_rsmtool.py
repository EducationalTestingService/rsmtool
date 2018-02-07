import warnings

from glob import glob
from os.path import basename, dirname, exists, join

from nose.tools import raises
from rsmtool.configuration_parser import ConfigurationParser

from rsmtool.test_utils import (check_file_output,
                                check_report,
                                check_scaled_coefficients,
                                check_subgroup_outputs,
                                check_generated_output,
                                check_consistency_files_exist,
                                do_run_experiment)

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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
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


def test_run_experiment_lars():

    # basic experiment with Lars model

    source = 'lars'
    experiment_id = 'Lars'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'skll'
    yield check_report, html_report


def test_run_experiment_lars_custom_objective():

    # experiment with lars model and a custom tuning objective

    source = 'lars-custom-objective'
    experiment_id = 'Lars_custom_objective'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'skll'
    yield check_report, html_report


def test_run_experiment_dummyregressor():

    # basic experiment with DummyRegressor model

    source = 'dummyregressor'
    experiment_id = 'DummyRegressor'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'skll'
    yield check_report, html_report


def test_run_experiment_dummyregressor_custom_objective():

    # experiment with dummyregressor model and a custom tuning objective

    source = 'dummyregressor-custom-objective'
    experiment_id = 'DummyRegressor_custom_objective'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'skll'
    yield check_report, html_report


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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'skll'
    yield check_report, html_report


def test_run_experiment_ridge_custom_objective():

    # experiment with ridge model and a custom tuning objective

    source = 'ridge-custom-objective'
    experiment_id = 'Ridge_custom_objective'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'skll'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'skll'
    yield check_report, html_report


def test_run_experiment_linearsvr_custom_objective():

    # experiment with LinearSVR model and a custom tuning objective

    source = 'linearsvr-custom-objective'
    experiment_id = 'LinearSVR_custom_objective'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'skll'
    yield check_report, html_report


def test_run_experiment_wls():

    # basic experiment with weighted least squares model

    source = 'wls'
    experiment_id = 'wls'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'skll'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_subgroup_outputs, output_dir, experiment_id, ['ITEM', 'QUESTION']
    yield check_report, html_report


def test_run_experiment_lr_with_id_with_leading_zeros():

    #  basic experiment with LinearRegression model including responses with
    #  candidate ids with leading zeros that should be kept in the outputs

    source = 'lr-with-id-with-leading-zeros'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_subgroup_outputs, output_dir, experiment_id, ['ITEM', 'QUESTION']
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
        html_report = join('test_outputs', source, 'report',
                           '{}_report.html'.format(experiment_id))

        csv_files = glob(join(output_dir, '*.csv'))
        for csv_file in csv_files:
            csv_filename = basename(csv_file)
            expected_csv_file = join(expected_output_dir, csv_filename)

            if exists(expected_csv_file):
                yield check_file_output, csv_file, expected_csv_file

        yield check_generated_output, csv_files, experiment_id, 'rsmtool'
        yield check_scaled_coefficients, source, experiment_id
        yield check_subgroup_outputs, output_dir, experiment_id, ['group_edge_cases']
        yield check_report, html_report


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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
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
            yield check_file_output, csv_file, expected_csv_file

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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


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


@raises(ValueError)
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


@raises(ValueError)
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


def test_run_experiment_lr_with_cfg():
    # basic experiment with a LinearRegression model

    source = 'lr-cfg'
    experiment_id = 'lr_cfg'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.cfg'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_with_object():
    # basic experiment with a LinearRegression model

    source = 'lr-object'
    experiment_id = 'lr_object'

    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))

    config_dict = {"train_file": "../../files/train.csv",
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
                   "description": "Using all features with an LinearRegression model."}

    config_parser = ConfigurationParser()
    config_parser.load_config_from_dict(config_dict)
    config_obj = config_parser.normalize_validate_and_process_config()
    config_obj = config_file

    do_run_experiment(source, experiment_id, config_obj)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_no_standardization():
    # basic experiment with a LinearRegression model,
    # with no standardization of features

    source = 'lr-no-standardization'
    experiment_id = 'lr_no_standardization'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_exclude_test_flags():
    # basic experiment with a LinearRegression model,
    # with only test flags removed

    source = 'lr-exclude-test-flags'
    experiment_id = 'lr_exclude_test_flags'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_exclude_train_and_test_flags():
    # basic experiment with a LinearRegression model,
    # with different train and test flags

    source = 'lr-exclude-train-and-test-flags'
    experiment_id = 'lr_exclude_train_and_test_flags'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_with_xlsx_output():
    # basic experiment with a LinearRegression model
    # with XLSX files as outputs

    source = 'lr-with-xlsx-output'
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

    xlsx_files = glob(join(output_dir, '*.xlsx'))
    for xlsx_file in xlsx_files:
        xlsx_filename = basename(xlsx_file)
        expected_xlsx_file = join(expected_output_dir, xlsx_filename)

        if exists(expected_xlsx_file):
            yield check_file_output, xlsx_file, expected_xlsx_file, 'xlsx'

    yield check_generated_output, xlsx_files, experiment_id, 'rsmtool', 'xlsx'
    yield check_scaled_coefficients, source, experiment_id, 'xlsx'
    yield check_report, html_report


def test_run_experiment_lr_with_tsv_output():
    # basic experiment with a LinearRegression model
    # with TSV files as outputs

    source = 'lr-with-tsv-output'
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

    tsv_files = glob(join(output_dir, '*.tsv'))
    for tsv_file in tsv_files:
        tsv_filename = basename(tsv_file)
        expected_tsv_file = join(expected_output_dir, tsv_filename)

        if exists(expected_tsv_file):
            yield check_file_output, tsv_file, expected_tsv_file, 'tsv'

    yield check_generated_output, tsv_files, experiment_id, 'rsmtool', 'tsv'
    yield check_scaled_coefficients, source, experiment_id, 'tsv'
    yield check_report, html_report


def test_run_experiment_lr_with_thumbnails():
    # basic experiment with a LinearRegression model
    # with thumbnail conversion

    source = 'lr-with-thumbnails'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_with_thumbnails_subgroups():

    # basic experiment with LinearRegression model but also including
    # subgroup analyses, with thumbnail conversion

    source = 'lr-with-thumbnails-subgroups'
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
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_subgroup_outputs, output_dir, experiment_id, ['L1']
    yield check_report, html_report
