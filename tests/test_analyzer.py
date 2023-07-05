import unittest
import warnings
from os.path import dirname, join

import numpy as np
import pandas as pd
from numpy.random import RandomState
from numpy.testing import assert_array_equal
from pandas.testing import assert_series_equal

from rsmtool.analyzer import Analyzer


class TestAnalyzer(unittest.TestCase):
    """Test class for Analyzer tests."""

    @classmethod
    def setUpClass(cls):
        cls.prng = RandomState(133)

        cls.df_features = pd.DataFrame(
            {
                "sc1": [1, 2, 3, 4, 1, 2, 3, 4, 1, 2],
                "f1": cls.prng.normal(0, 1, 10),
                "f2": cls.prng.normal(1, 0.1, 10),
                "f3": cls.prng.normal(2, 0.1, 10),
                "group": ["group1"] * 10,
            },
            index=range(0, 10),
        )

        cls.df_features_same_score = cls.df_features.copy()
        cls.df_features_same_score["sc1"] = [3] * 10

        cls.df_features_with_groups = cls.df_features.copy()
        cls.df_features_with_groups["group"] = ["group1"] * 5 + ["group2"] * 5

        cls.df_features_with_groups_and_length = cls.df_features_with_groups.copy()
        cls.df_features_with_groups_and_length["length"] = cls.prng.normal(50, 250, 10)

        cls.human_scores = pd.Series(cls.prng.randint(1, 5, size=10))
        cls.system_scores = pd.Series(cls.prng.random_sample(10) * 5)
        cls.same_human_scores = pd.Series([3] * 10)

        # get the directory containing the tests
        cls.test_dir = dirname(__file__)

    def test_correlation_helper(self):
        # test that there are no nans for data frame with 10 values
        retval = Analyzer.correlation_helper(self.df_features, "sc1", "group")
        self.assertEqual(retval[0].isnull().values.sum(), 0)
        self.assertEqual(retval[1].isnull().values.sum(), 0)

    def test_correlation_helper_for_data_with_one_row(self):
        # this should return two data frames with nans
        retval = Analyzer.correlation_helper(self.df_features[:1], "sc1", "group")
        self.assertEqual(retval[0].isnull().values.sum(), 3)
        self.assertEqual(retval[1].isnull().values.sum(), 3)

    def test_correlation_helper_for_data_with_two_rows(self):
        # this should return 1/-1 for marginal correlations and nans for
        # partial correlations
        retval = Analyzer.correlation_helper(self.df_features[:2], "sc1", "group")
        self.assertEqual(abs(retval[0].values).sum(), 3)
        self.assertEqual(retval[1].isnull().values.sum(), 3)

    def test_correlation_helper_for_data_with_three_rows(self):
        # this should compute marginal correlations but return Nans for
        # partial correlations
        retval = Analyzer.correlation_helper(self.df_features[:3], "sc1", "group")
        self.assertEqual(retval[0].isnull().values.sum(), 0)
        self.assertEqual(retval[1].isnull().values.sum(), 3)

    def test_correlation_helper_for_data_with_four_rows(self):
        # this should compute marginal correlations and return a unity
        # matrix for partial correlations
        # it should also raise a UserWarning
        with warnings.catch_warnings(record=True) as warning_list:
            retval = Analyzer.correlation_helper(self.df_features[:4], "sc1", "group")
        self.assertEqual(retval[0].isnull().values.sum(), 0)
        self.assertAlmostEqual(np.abs(retval[1].values).sum(), 0.9244288637889855)
        assert issubclass(warning_list[-1].category, UserWarning)

    def test_correlation_helper_for_data_with_groups(self):
        retval = Analyzer.correlation_helper(self.df_features_with_groups, "sc1", "group")
        self.assertEqual(len(retval[0]), 2)
        self.assertEqual(len(retval[1]), 2)

    def test_correlation_helper_for_one_group_with_one_row(self):
        # this should return a data frames with nans for group with 1 row
        retval = Analyzer.correlation_helper(self.df_features_with_groups[:6], "sc1", "group")
        self.assertEqual(len(retval[0]), 2)
        self.assertEqual(len(retval[1]), 2)
        self.assertEqual(retval[0].isnull().values.sum(), 3)

    def test_correlation_helper_for_groups_and_length(self):
        retval = Analyzer.correlation_helper(
            self.df_features_with_groups_and_length, "sc1", "group", include_length=True
        )
        for df in retval:
            self.assertEqual(len(df), 2)
            self.assertEqual(len(df.columns), 3)

    def test_correlation_helper_for_group_with_one_row_and_length(self):
        # this should return a data frames with nans for group with 1 row
        retval = Analyzer.correlation_helper(
            self.df_features_with_groups_and_length[:6],
            "sc1",
            "group",
            include_length=True,
        )
        for df in retval:
            self.assertEqual(len(df), 2)
            self.assertEqual(len(df.columns), 3)

    def test_that_correlation_helper_works_for_data_with_the_same_human_score(self):
        # this test should raise UserWarning because the determinant is very close to
        # zero. It also raises Runtime warning because
        # variance of human scores is 0.
        with warnings.catch_warnings(record=True) as warning_list:
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            retval = Analyzer.correlation_helper(self.df_features_same_score, "sc1", "group")
            self.assertEqual(retval[0].isnull().values.sum(), 3)
            self.assertEqual(retval[1].isnull().values.sum(), 3)
            assert issubclass(warning_list[-1].category, UserWarning)

    def test_that_metrics_helper_works_for_data_with_one_row(self):
        # There should be NaNs for SMD, correlations and both sds
        # note that we will get a value for QWK since we are
        # dividing by N and not N-1
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            evals = Analyzer.metrics_helper(self.human_scores[0:1], self.system_scores[0:1])
            self.assertEqual(evals.isnull().values.sum(), 5)

    def test_that_metrics_helper_works_for_data_with_the_same_label(self):
        # There should be NaNs for correlation and SMD.
        # Note that for a dataset with a single response
        # kappas will be 0 or 1
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            evals = Analyzer.metrics_helper(self.same_human_scores, self.system_scores)
            self.assertEqual(evals.isnull().values.sum(), 2)

    def test_metrics_helper_population_sds(self):
        df_new_features = pd.read_csv(join(self.test_dir, "data", "files", "train.csv"))
        # compute the metrics when not specifying the population SDs
        computed_metrics1 = Analyzer.metrics_helper(
            df_new_features["score"], df_new_features["score2"]
        )
        expected_metrics1 = pd.Series(
            {
                "N": 500.0,
                "R2": 0.65340566606389394,
                "RMSE": 0.47958315233127197,
                "SMD": 0.03679030063229779,
                "adj_agr": 100.0,
                "corr": 0.82789026370069529,
                "exact_agr": 77.0,
                "h_max": 6.0,
                "h_mean": 3.4199999999999999,
                "h_min": 1.0,
                "h_sd": 0.81543231461565147,
                "kappa": 0.6273493195074531,
                "sys_max": 6.0,
                "sys_mean": 3.4500000000000002,
                "sys_min": 1.0,
                "sys_sd": 0.81782496620652367,
                "wtkappa": 0.8273273273273274,
            }
        )

        # and now compute them specifying the population SDs
        computed_metrics2 = Analyzer.metrics_helper(
            df_new_features["score"],
            df_new_features["score2"],
            population_human_score_sd=0.5,
            population_system_score_sd=0.4,
            smd_method="williamson",
        )
        # the only number that should change is the SMD
        expected_metrics2 = expected_metrics1.copy()
        expected_metrics2["SMD"] = 0.066259

        assert_series_equal(computed_metrics1.sort_index(), expected_metrics1.sort_index())
        assert_series_equal(computed_metrics2.sort_index(), expected_metrics2.sort_index())

    def test_metrics_helper_zero_system_sd(self):
        human_scores = [1, 3, 4, 2, 3, 1, 3, 4, 2, 1]
        system_score = [2.54] * 10
        computed_metrics1 = Analyzer.metrics_helper(human_scores, system_score)
        expected_metrics1 = pd.Series(
            {
                "N": 10,
                "R2": -0.015806451612903283,
                "RMSE": 1.122319027727856,
                "SMD": 0.11927198519188371,
                "adj_agr": 50.0,
                "corr": None,
                "exact_agr": 0,
                "h_max": 4,
                "h_mean": 2.4,
                "h_min": 1.0,
                "h_sd": 1.1737877907772674,
                "kappa": 0,
                "sys_max": 2.54,
                "sys_mean": 2.54,
                "sys_min": 2.54,
                "sys_sd": 0,
                "wtkappa": 0,
            }
        )
        # now compute DSM
        computed_metrics2 = Analyzer.metrics_helper(
            human_scores, system_score, use_diff_std_means=True
        )

        # the only number that should change is the SMD
        expected_metrics2 = expected_metrics1.copy()
        expected_metrics2.drop("SMD", inplace=True)
        expected_metrics2["DSM"] = None
        assert_series_equal(
            computed_metrics1.sort_index(),
            expected_metrics1.sort_index(),
            check_dtype=False,
        )
        assert_series_equal(
            computed_metrics2.sort_index(),
            expected_metrics2.sort_index(),
            check_dtype=False,
        )

    def test_compute_pca_less_samples_than_features(self):
        # test pca when we have less samples than
        # features. In this case the number of components
        # equals to the number of samples.
        dfs = []
        # to avoid inserting too many columns,
        # we create a list of data frames and then
        # concatenate them together
        for i in range(1, 101):
            dfs.append(pd.DataFrame({i: pd.Series(range(50)) * i}))
        df = pd.concat(dfs, axis=1)
        (components, variance) = Analyzer.compute_pca(df, df.columns)
        self.assertEqual(len(components.columns), 50)
        self.assertEqual(len(variance.columns), 50)

    def test_compute_disattenuated_correlations_single_human(self):
        hm_corr = pd.Series([0.9, 0.8, 0.6], index=["raw", "raw_trim", "raw_trim_round"])
        hh_corr = pd.Series([0.81], index=[""])
        df_dis_corr = Analyzer.compute_disattenuated_correlations(hm_corr, hh_corr)
        self.assertEqual(len(df_dis_corr), 3)
        self.assertEqual(df_dis_corr.loc["raw", "corr_disattenuated"], 1.0)

    def test_compute_disattenuated_correlations_matching_human(self):
        hm_corr = pd.Series([0.9, 0.4, 0.6], index=["All data", "GROUP1", "GROUP2"])
        hh_corr = pd.Series([0.81, 0.64, 0.36], index=["All data", "GROUP1", "GROUP2"])
        df_dis_corr = Analyzer.compute_disattenuated_correlations(hm_corr, hh_corr)
        self.assertEqual(len(df_dis_corr), 3)
        assert_array_equal(df_dis_corr["corr_disattenuated"], [1.0, 0.5, 1.0])

    def test_compute_disattenuated_correlations_single_matching_human(self):
        hm_corr = pd.Series([0.9, 0.4, 0.6], index=["All data", "GROUP1", "GROUP2"])
        hh_corr = pd.Series([0.81], index=["All data"])
        df_dis_corr = Analyzer.compute_disattenuated_correlations(hm_corr, hh_corr)
        self.assertEqual(len(df_dis_corr), 3)
        assert_array_equal(df_dis_corr["corr_disattenuated"], [1.0, np.nan, np.nan])

    def test_compute_disattenuated_correlations_mismatched_indices(self):
        hm_corr = pd.Series([0.9, 0.6], index=["All data", "GROUP2"])
        hh_corr = pd.Series([0.81, 0.64], index=["All data", "GROUP1"])
        df_dis_corr = Analyzer.compute_disattenuated_correlations(hm_corr, hh_corr)
        self.assertEqual(len(df_dis_corr), 3)
        assert_array_equal(df_dis_corr["corr_disattenuated"], [1.0, np.nan, np.nan])

    def test_compute_disattenuated_correlations_negative_human(self):
        hm_corr = pd.Series([0.9, 0.8], index=["All data", "GROUP1"])
        hh_corr = pd.Series([-0.03, 0.64], index=["All data", "GROUP1"])
        df_dis_corr = Analyzer.compute_disattenuated_correlations(hm_corr, hh_corr)
        self.assertEqual(len(df_dis_corr), 2)
        assert_array_equal(df_dis_corr["corr_disattenuated"], [np.nan, 1.0])
