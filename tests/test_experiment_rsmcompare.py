import os

from os.path import join

from nose.tools import raises
from parameterized import param, parameterized

from rsmtool.test_utils import (check_run_comparison,
                                do_run_comparison,
                                rsmtool_test_dir)

TEST_DIR = os.environ.get('TESTDIR', None)


# set this to False to disable auto-updating of all experiment
# tests contained in this file via `update_files.py`.
# TODO: re-enable this once we start saving rsmcompare outputs
_AUTO_UPDATE = False


@parameterized([
    param('lr-self-compare', 'lr_subgroups_vs_lr_subgroups_report'),
    param('lr-different-compare', 'lr_baseline_vs_lr_with_FEATURE8_and_zero_scores_report'),
    param('lr-self-compare-with-h2', 'lr_with_h2_vs_lr_with_h2_report'),
    param('lr-self-compare-with-custom-order', 'lr_subgroups_vs_lr_subgroups_report'),
    param('lr-self-compare-with-chosen-sections', 'lr_subgroups_vs_lr_subgroups_report'),
    param('lr-self-compare-with-custom-sections-and-custom-order', 'lr_subgroups_vs_lr_subgroups_report'),
    param('lr-self-compare-with-thumbnails', 'lr_subgroups_vs_lr_subgroups_report'),
    param('linearsvr-self-compare', 'LinearSVR_vs_LinearSVR_report'),
    param('lr-eval-self-compare', 'lr_eval_with_h2_vs_lr_eval_with_h2_report'),
    param('lr-eval-tool-compare', 'lr_with_h2_vs_lr_eval_with_h2_report'),
    param('lr-self-compare-different-format', 'lr_subgroups_vs_lr_subgroups_report'),
    param('lr-self-compare-with-subgroups-and-h2', 'lr-subgroups-with-h2_vs_lr-subgroups-with-h2_report'),
    param('lr-self-compare-with-subgroups-and-edge-cases', 'lr-subgroups-with-edge-cases_vs_lr-subgroups-with-edge-cases_report')
])
def test_run_experiment_parameterized(*args, **kwargs):
    if TEST_DIR:
        kwargs['given_test_dir'] = TEST_DIR
    check_run_comparison(*args, **kwargs)


@raises(FileNotFoundError)
def test_run_experiment_lr_compare_wrong_directory():

    # basic rsmcompare experiment comparing a LinearRegression
    # experiment to itself
    source = 'lr-self-compare-wrong-directory'
    config_file = join(rsmtool_test_dir,
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
    config_file = join(rsmtool_test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmcompare.json')
    do_run_comparison(source, config_file)
