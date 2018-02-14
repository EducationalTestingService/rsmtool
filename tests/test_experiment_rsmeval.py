from glob import glob
from os.path import basename, exists, join

from nose.tools import raises
from parameterized import param, parameterized

from rsmtool.test_utils import (check_file_output,
                                check_report,
                                check_run_evaluation,
                                do_run_evaluation,
                                test_dir)


@parameterized([
    param('lr-eval', 'lr_evaluation'),
    param('lr-eval-with-scaling', 'lr_evaluation_with_scaling'),
    param('lr-eval-exclude-listwise', 'lr_eval_exclude_listwise'),
    param('lr-eval-exclude-flags', 'lr_eval_exclude_flags'),
    param('lr-eval-with-missing-scores', 'lr_eval_with_missing_scores'),
    param('lr-eval-with-missing-data', 'lr_eval_with_missing_data'),
    param('lr-eval-with-custom-order', 'lr_eval_with_custom_order'),
    param('lr-eval-with-custom-sections', 'lr_eval_with_custom_sections'),
    param('lr-eval-with-custom-sections-and-order', 'lr_eval_with_custom_sections_and_order'),
    param('lr-eval-tsv-input-files', 'lr_eval_tsv_input_files'),
    param('lr-eval-xlsx-input-files', 'lr_eval_xlsx_input_files'),
    param('lr-eval-with-tsv-output', 'lr_eval_with_tsv_output'),
    param('lr-eval-with-xlsx-output', 'lr_eval_with_xlsx_output'),
    param('lr-eval-with-h2', 'lr_eval_with_h2', consistency=True),
    param('lr-eval-with-h2-named-sc1', 'lr_eval_with_h2_named_sc1', consistency=True),
    param('lr-eval-with-scaling-and-h2-keep-zeros', 'lr_eval_with_scaling_and_h2_keep_zeros', consistency=True),
])
def test_run_experiment_parameterized(*args, **kwargs):
    check_run_evaluation(*args, **kwargs)


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


@raises(FileNotFoundError)
def test_run_experiment_lr_eval_wrong_path():

    # basic rsmeval experiment with wrong path to the
    # predictions file

    source = 'lr-eval-with-wrong-path'
    experiment_id = 'lr_eval_with_h2'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_evaluation(source, experiment_id, config_file)
