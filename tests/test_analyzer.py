import warnings

from os.path import dirname, join

import numpy as np
import pandas as pd

from nose.tools import (assert_almost_equal, assert_equal)
from numpy.random import RandomState
from pandas.util.testing import assert_series_equal
from numpy.testing import assert_array_equal

from rsmtool.analyzer import Analyzer


class TestAnalyzer:

    def setUp(self):

        self.prng = RandomState(133)

        self.df_features = pd.DataFrame({'sc1': [1, 2, 3, 4, 1, 2, 3, 4, 1, 2],
                                         'f1': self.prng.normal(0, 1, 10),
                                         'f2': self.prng.normal(1, 0.1, 10),
                                         'f3': self.prng.normal(2, 0.1, 10),
                                         'group': ['group1'] * 10},
                                        index=range(0, 10))

        self.df_features_same_score = self.df_features.copy()
        self.df_features_same_score[['sc1']] = [3] * 10

        self.human_scores = pd.Series(self.prng.randint(1, 5, size=10))
        self.system_scores = pd.Series(self.prng.random_sample(10) * 5)
        self.same_human_scores = pd.Series([3] * 10)

        # get the directory containing the tests
        self.test_dir = dirname(__file__)

    def test_correlation_helper(self):

        # test that there are no nans for data frame with 10 values
        retval = Analyzer.correlation_helper(self.df_features, 'sc1', 'group')
        assert_equal(retval[0].isnull().values.sum(), 0)
        assert_equal(retval[1].isnull().values.sum(), 0)

    def test_that_correlation_helper_works_for_data_with_one_row(self):
        # this should return two data frames with nans
        # we expect a runtime warning here so let's suppress it
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            retval = Analyzer.correlation_helper(self.df_features[:1], 'sc1', 'group')
            assert_equal(retval[0].isnull().values.sum(), 3)
            assert_equal(retval[1].isnull().values.sum(), 3)

    def test_that_correlation_helper_works_for_data_with_two_rows(self):
        # this should return 1/-1 for marginal correlations and nans for
        # partial correlations
        retval = Analyzer.correlation_helper(self.df_features[:2], 'sc1', 'group')
        assert_equal(abs(retval[0].values).sum(), 3)
        assert_equal(retval[1].isnull().values.sum(), 3)

    def test_that_correlation_helper_works_for_data_with_three_rows(self):
        # this should compute marginal correlations but return Nans for
        # partial correlations
        retval = Analyzer.correlation_helper(self.df_features[:3], 'sc1', 'group')
        assert_equal(retval[0].isnull().values.sum(), 0)
        assert_equal(retval[1].isnull().values.sum(), 3)

    def test_that_correlation_helper_works_for_data_with_four_rows(self):
        # this should compute marginal correlations and return a unity
        # matrix for partial correlations
        retval = Analyzer.correlation_helper(self.df_features[:4], 'sc1', 'group')
        assert_equal(retval[0].isnull().values.sum(), 0)
        assert_almost_equal(abs(retval[1].values).sum(), 3)

    def test_that_correlation_helper_works_for_data_with_the_same_label(self):

        # this should return two data frames with nans
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            retval = Analyzer.correlation_helper(self.df_features_same_score, 'sc1', 'group')
            assert_equal(retval[0].isnull().values.sum(), 3)
            assert_equal(retval[1].isnull().values.sum(), 3)

    def test_that_metrics_helper_works_for_data_with_one_row(self):
        # There should be NaNs for SMD, correlations and both sds
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            evals = Analyzer.metrics_helper(self.human_scores[0:1],
                                            self.system_scores[0:1])
            assert_equal(evals.isnull().values.sum(), 5)

    def test_that_metrics_helper_works_for_data_with_the_same_label(self):
        # There should be NaNs for correlation and SMD.
        # Note that for a dataset with a single response
        # kappas will be 0 or 1
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            evals = Analyzer.metrics_helper(self.same_human_scores,
                                            self.system_scores)
            print(evals)
            assert_equal(evals.isnull().values.sum(), 2)

    def test_metrics_helper_population_sds(self):
        df_new_features = pd.read_csv(join(self.test_dir, 'data', 'files', 'train.csv'))
        # compute the metrics when not specifying the population SDs
        computed_metrics1 = Analyzer.metrics_helper(df_new_features['score'],
                                                    df_new_features['score2'])
        expected_metrics1 = pd.Series({'N': 500.0,
                                       'R2': 0.65340566606389394,
                                       'RMSE': 0.47958315233127197,
                                       'SMD': 0.03679030063229779,
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
                                       'wtkappa': 0.8276724397975914})

        # and now compute them specifying the population SDs
        computed_metrics2 = Analyzer.metrics_helper(df_new_features['score'],
                                                    df_new_features['score2'],
                                                    population_human_score_sd=0.5,
                                                    population_system_score_sd=0.4,
                                                    smd_method='williamson')
        # the only number that should change is the SMD
        expected_metrics2 = expected_metrics1.copy()
        expected_metrics2['SMD'] = 0.066259

        assert_series_equal(computed_metrics1.sort_index(), expected_metrics1.sort_index())
        assert_series_equal(computed_metrics2.sort_index(), expected_metrics2.sort_index())

    def test_compute_pca_less_components_than_features(self):
        # test pca when we have less components than features
        df = pd.DataFrame({'a': range(100)})
        for i in range(100):
            df[i] = df['a'] * i
        (components, variance) = Analyzer.compute_pca(df, df.columns)
        assert_equal(len(components.columns), 100)
        assert_equal(len(variance.columns), 100)

    def test_compute_disattenuated_correlations_single_human(self):
        hm_corr = pd.Series([0.9, 0.8, 0.6],
                            index=['raw', 'raw_trim', 'raw_trim_round'])
        hh_corr = pd.Series([0.81], index=[''])
        df_dis_corr = Analyzer.compute_disattenuated_correlations(hm_corr,
                                                                  hh_corr)
        assert_equal(len(df_dis_corr), 3)
        assert_equal(df_dis_corr.loc['raw', 'corr_disattenuated'], 1.0)

    def test_compute_disattenuated_correlations_matching_human(self):
        hm_corr = pd.Series([0.9, 0.4, 0.6],
                            index=['All data', 'GROUP1', 'GROUP2'])
        hh_corr = pd.Series([0.81, 0.64, 0.36],
                            index=['All data', 'GROUP1', 'GROUP2'])
        df_dis_corr = Analyzer.compute_disattenuated_correlations(hm_corr,
                                                                  hh_corr)
        assert_equal(len(df_dis_corr), 3)
        assert_array_equal(df_dis_corr['corr_disattenuated'], [1.0, 0.5, 1.0])

    def test_compute_disattenuated_correlations_single_matching_human(self):
        hm_corr = pd.Series([0.9, 0.4, 0.6],
                            index=['All data', 'GROUP1', 'GROUP2'])
        hh_corr = pd.Series([0.81],
                            index=['All data'])
        df_dis_corr = Analyzer.compute_disattenuated_correlations(hm_corr,
                                                                  hh_corr)
        assert_equal(len(df_dis_corr), 3)
        assert_array_equal(df_dis_corr['corr_disattenuated'], [1.0, np.nan, np.nan])

    def test_compute_disattenuated_correlations_mismatched_indices(self):
        hm_corr = pd.Series([0.9, 0.6],
                            index=['All data', 'GROUP2'])
        hh_corr = pd.Series([0.81, 0.64],
                            index=['All data', 'GROUP1'])
        df_dis_corr = Analyzer.compute_disattenuated_correlations(hm_corr,
                                                                  hh_corr)
        assert_equal(len(df_dis_corr), 3)
        assert_array_equal(df_dis_corr['corr_disattenuated'], [1.0, np.nan, np.nan])

    def test_compute_disattenuated_correlations_negative_human(self):
        hm_corr = pd.Series([0.9, 0.8],
                            index=['All data', 'GROUP1'])
        hh_corr = pd.Series([-0.03, 0.64],
                            index=['All data', 'GROUP1'])
        df_dis_corr = Analyzer.compute_disattenuated_correlations(hm_corr,
                                                                  hh_corr)
        assert_equal(len(df_dis_corr), 2)
        assert_array_equal(df_dis_corr['corr_disattenuated'], [np.nan, 1.0])
