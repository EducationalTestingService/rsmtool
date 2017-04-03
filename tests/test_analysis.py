import warnings

from os.path import dirname, join

import pandas as pd

from nose.tools import (assert_almost_equal, assert_equal)
from numpy.random import RandomState
from pandas.util.testing import assert_series_equal

from rsmtool.analysis import (compute_pca,
                              correlation_helper,
                              metrics_helper)

prng = RandomState(133)
df_features = pd.DataFrame({'sc1': [1, 2, 3, 4, 1, 2, 3, 4, 1, 2],
                            'f1': prng.normal(0, 1, 10),
                            'f2': prng.normal(1, 0.1, 10),
                            'f3': prng.normal(2, 0.1, 10),
                            'group': ['group1']*10},
                             index  = range(0, 10))

df_features_same_score = df_features.copy()
df_features_same_score[['sc1']] = [3]*10

human_scores = pd.Series(prng.randint(1, 5, size=10))
system_scores = pd.Series(prng.random_sample(10)*5)
same_human_scores = pd.Series([3]*10)

# get the directory containing the tests
test_dir = dirname(__file__)

def test_correlation_helper():
    # test that there are no nans for data frame with 10 values
    retval = correlation_helper(df_features, 'sc1', 'group')
    assert_equal(retval[0].isnull().values.sum(), 0)
    assert_equal(retval[1].isnull().values.sum(), 0)


def test_that_correlation_helper_works_for_data_with_one_row():
    # this should return two data frames with nans
    # we expect a runtime warning here so let's suppress it
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        retval = correlation_helper(df_features[:1], 'sc1', 'group')
        assert_equal(retval[0].isnull().values.sum(), 3)
        assert_equal(retval[1].isnull().values.sum(), 3)


def test_that_correlation_helper_works_for_data_with_two_rows():
    # this should return 1/-1 for marginal correlations and nans for
    # partial correlations
    retval = correlation_helper(df_features[:2], 'sc1', 'group')
    assert_equal(abs(retval[0].values).sum(), 3)
    assert_equal(retval[1].isnull().values.sum(), 3)



def test_that_correlation_helper_works_for_data_with_three_rows():
    # this should compute marginal correlations but return Nans for
    # partial correlations
    retval = correlation_helper(df_features[:3], 'sc1', 'group')
    assert_equal(retval[0].isnull().values.sum(), 0)
    assert_equal(retval[1].isnull().values.sum(), 3)


def test_that_correlation_helper_works_for_data_with_four_rows():
    # this should compute marginal correlations and return a unity
    # matrix for partial correlations
    retval = correlation_helper(df_features[:4], 'sc1', 'group')
    assert_equal(retval[0].isnull().values.sum(), 0)
    assert_almost_equal(abs(retval[1].values).sum(), 3)


def test_that_correlation_helper_works_for_data_with_the_same_label():
    # this should return two data frames with nans
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        retval = correlation_helper(df_features_same_score, 'sc1', 'group')
        assert_equal(retval[0].isnull().values.sum(), 3)
        assert_equal(retval[1].isnull().values.sum(), 3)


def test_that_metrics_helper_works_for_data_with_one_row():
    # There should be NaNs for SMD, correlations and both sds
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        evals = metrics_helper(human_scores[0:1], system_scores[0:1])
        assert_equal(evals.isnull().values.sum(), 4)


def test_that_metrics_helper_works_for_data_with_the_same_label():
    # There should be NaNs for correlation.
    # Note that for a dataset with a single response
    # kappas will be 0 or 1
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        evals = metrics_helper(same_human_scores, system_scores)
        assert_equal(evals.isnull().values.sum(), 1)


def test_metrics_helper_population_sds():
    df_features = pd.read_csv(join(test_dir, 'data', 'files', 'train.csv'))
    # compute the metrics when not specifying the population SDs
    computed_metrics1 = metrics_helper(df_features['score'], df_features['score2'])
    expected_metrics1 = pd.Series({'N': 500.0,
                                  'R2': 0.65340566606389394,
                                  'RMSE': 0.47958315233127197,
                                  'SMD': 0.036736365006090885,
                                  'adj_agr': 100.0,
                                  'corr': 0.82789026370069529,
                                  'exact_agr': 77.0,
                                  'h_max': 6.0,
                                  'h_mean': 3.4199999999999999,
                                  'h_min': 1.0,
                                  'h_sd': 0.81543231461565147,
                                  'kappa': 0.6273493195074531,
                                  'sys_max': 6.0,
                                  'sys_mean': 3.4500000000000002,
                                  'sys_min': 1.0,
                                  'sys_sd': 0.81782496620652367,
                                  'wtkappa': 0.82732732732732728})
    # and now compute them specifying the population SDs
    computed_metrics2 = metrics_helper(df_features['score'],
                                       df_features['score2'],
                                       population_human_score_sd=0.5,
                                       population_system_score_sd=0.4)
    # the only number that should change is the SMD
    expected_metrics2 = expected_metrics1.copy()
    expected_metrics2['SMD'] = 0.066259

    assert_series_equal(computed_metrics1, expected_metrics1)
    assert_series_equal(computed_metrics2, expected_metrics2)


def test_compute_pca_less_components_than_features():
    # test pca when we have less components than features
    df = pd.DataFrame({'a':range(100)})
    for i in range(100):
        df[i] = df['a']*i
    (components, variance) = compute_pca(df, df.columns)
    assert_equal(len(components.columns), 100)
    assert_equal(len(variance.columns), 100)

