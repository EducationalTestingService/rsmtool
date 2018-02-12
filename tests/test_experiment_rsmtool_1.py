import warnings

from glob import glob
from os.path import basename, dirname, exists, join

from nose.tools import raises

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


# -- 3 --
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
