import warnings

import pandas as pd

from nose.tools import (assert_almost_equal, assert_equal)
from numpy.random import RandomState

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


def test_compute_pca_less_components_than_features():
    # test pca when we have less components than features
    df = pd.DataFrame({'a':range(100)})
    for i in range(100):
        df[i] = df['a']*i
    (components, variance) = compute_pca(df, df.columns)
    assert_equal(len(components.columns), 100)
    assert_equal(len(variance.columns), 100)

