import os

from os.path import join

from nose.tools import raises
from parameterized import param, parameterized

from rsmtool.test_utils import (check_run_experiment,
                                do_run_experiment)

# allow test directory to be set via an environment variable
# which is needed for package testing
TEST_DIR = os.environ.get('TESTDIR', None)
if TEST_DIR:
    rsmtool_test_dir = TEST_DIR
else:
    from rsmtool.test_utils import rsmtool_test_dir


@parameterized([
    param('lr-with-h2-include-zeros', 'lr_with_h2_include_zeros', consistency=True),
    param('lr-with-h2-and-length', 'lr_with_h2_and_length', consistency=True),
    param('lr-with-h2-named-sc1', 'lr_with_h2_named_sc1', consistency=True),
    param('lars', 'Lars', skll=True),
    param('lars-custom-objective', 'Lars_custom_objective', skll=True),
    param('logistic-regression', 'LogisticRegression', skll=True),
    param('logistic-regression-custom-objective', 'LogisticRegression_custom_objective', skll=True),
    param('logistic-regression-expected-scores', 'LogisticRegression_expected_scores', skll=True),
    param('svc', 'SVC', skll=True),
    param('svc-custom-objective', 'SVC_custom_objective', skll=True),
    param('svc-expected-scores', 'SVC_expected_scores', skll=True),
    param('dummyregressor', 'DummyRegressor', skll=True),
    param('dummyregressor-custom-objective', 'DummyRegressor_custom_objective', skll=True),
    param('ridge', 'Ridge', skll=True),
    param('ridge-custom-objective', 'Ridge_custom_objective', skll=True),
    param('linearsvr', 'LinearSVR', skll=True),
    param('linearsvr-custom-objective', 'LinearSVR_custom_objective', skll=True),
    param('wls', 'wls', skll=True),  # treat this as SKLL since we don't want to test coefficients
    param('rebalancedlr', 'rebalancedlr'),
    param('lassofixedlambdathenlr', 'lassofixedlambdathenlr'),
    param('positivelassocvthenlr', 'positivelassocvthenlr'),
    param('nnlr', 'nnlr'),
    param('nnlr_iterative', 'nnlr_iterative'),
    param('lassofixedlambdathennnlr', 'lassofixedlambdathennnlr'),
    param('lassofixedlambda', 'lassofixedlambda'),
    param('positivelassocv', 'positivelassocv'),
    param('equalweightslr', 'equalweightslr'),
    param('lr-with-length-string', 'lr_with_length_string')
])
def test_run_experiment_parameterized(*args, **kwargs):
    if TEST_DIR:
        kwargs['given_test_dir'] = TEST_DIR
    check_run_experiment(*args, **kwargs)


@raises(ValueError)
def test_run_experiment_empwtdropneg():

    # rsmtool experiment with no longer supported empWtDropNeg model
    source = 'empwtdropneg'
    experiment_id = 'empWtDropNeg'
    config_file = join(rsmtool_test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_requested_feature_zero_sd():

    # rsmtool experiment when a requested feature has zero sd
    source = 'lr-with-requested-feature-with-zero-sd'
    experiment_id = 'lr_with_requested_feature_with_zero_sd'
    config_file = join(rsmtool_test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)

