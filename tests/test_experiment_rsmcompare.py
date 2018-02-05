from os.path import dirname, join

from nose.tools import raises

from rsmtool.test_utils import (check_report,
                                do_run_comparison)

# get the directory containing the tests
test_dir = dirname(__file__)


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

    html_report = join('test_outputs', source,
                       'lr_baseline_vs_lr_with_FEATURE8_and_zero_scores_report.html')
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


def test_run_experiment_lr_compare_different_format():

    # basic rsmcompare experiment comparing a LinearRegression
    # experiment to itself, with TSVs in output directory

    source = 'lr-self-compare-different-format'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmcompare.json')
    do_run_comparison(source, config_file)

    html_report = join('test_outputs', source, 'lr_subgroups_vs_lr_subgroups_report.html')
    yield check_report, html_report


def test_run_experiment_lr_compare_with_subgroups_and_edge_cases():

    # basic rsmcompare experiment comparing a LinearRegression
    # experiment to itself with in case where subgroups have
    # edge cases (such 1 or 2 cases per subgroup or the same score)
    
    source = 'lr-self-compare-with-subgroups-and-edge-cases'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmcompare.json')
    do_run_comparison(source, config_file)

    html_report = join('test_outputs', source, 'lr-subgroups-with-edge-cases_vs_lr-subgroups-with-edge-cases.html')
    yield check_report, html_report

