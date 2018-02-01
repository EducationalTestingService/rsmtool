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
                                do_run_experiment,
                                do_run_evaluation,
                                do_run_prediction,
                                do_run_comparison,
                                do_run_summary)

# get the directory containing the tests
test_dir = dirname(__file__)

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
            yield check_file_output, csv_file, expected_csv_file

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
            yield check_file_output, csv_file, expected_csv_file

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
            yield check_file_output, csv_file, expected_csv_file

    yield check_consistency_files_exist, csv_files, experiment_id
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
            yield check_file_output, csv_file, expected_csv_file

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
            yield check_file_output, csv_file, expected_csv_file

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
            yield check_file_output, csv_file, expected_csv_file

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
            yield check_file_output, csv_file, expected_csv_file

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
            yield check_file_output, csv_file, expected_csv_file

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
            yield check_file_output, csv_file, expected_csv_file

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
            yield check_file_output, csv_file, expected_csv_file

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
            yield check_file_output, csv_file, expected_csv_file

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
            yield check_file_output, csv_file, expected_csv_file

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
            yield check_file_output, csv_file, expected_csv_file

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
            yield check_file_output, csv_file, expected_csv_file

    yield check_report, html_report


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


def test_run_experiment_lr_eval_with_cfg():

    # basic evaluation experiment using rsmeval

    source = 'lr-eval-cfg'
    experiment_id = 'lr_eval_cfg'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.cfg'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)

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


def test_run_experiment_lr_eval_with_tsv_output():

    # basic evaluation experiment using rsmeval
    # output in TSV format

    source = 'lr-eval-with-tsv-output'
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

    tsv_files = glob(join(output_dir, '*.tsv'))
    for tsv_file in tsv_files:
        tsv_filename = basename(tsv_file)
        expected_tsv_file = join(expected_output_dir, tsv_filename)

        if exists(expected_tsv_file):
            yield check_file_output, tsv_file, expected_tsv_file, 'tsv'

    yield check_report, html_report


def test_run_experiment_lr_eval_with_xlsx_output():

    # basic evaluation experiment using rsmeval
    # output in XLSX format

    source = 'lr-eval-with-xlsx-output'
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

    xlsx_files = glob(join(output_dir, '*.xlsx'))
    for xlsx_file in xlsx_files:
        xlsx_filename = basename(xlsx_file)
        expected_xlsx_file = join(expected_output_dir, xlsx_filename)

        if exists(expected_xlsx_file):
            yield check_file_output, xlsx_file, expected_xlsx_file, 'xlsx'

    yield check_report, html_report

