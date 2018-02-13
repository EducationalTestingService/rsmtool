from os.path import join

from nose.tools import raises
from parameterized import param, parameterized

from rsmtool.test_utils import (check_run_experiment,
                                do_run_experiment,
                                test_dir)


@parameterized([
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
    param('lassofixedlambdathennnlr', 'lassofixedlambdathennnlr'),
    param('lassofixedlambda', 'lassofixedlambda'),
    param('positivelassocv', 'positivelassocv'),
    param('equalweightslr', 'equalweightslr')
])
def test_run_experiment_parameterized(*args, **kwargs):
    check_run_experiment(*args, **kwargs)


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
