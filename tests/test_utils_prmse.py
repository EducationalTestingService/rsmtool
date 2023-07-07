import os
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from rsmtool.utils.prmse import (
    get_n_human_scores,
    get_true_score_evaluations,
    mse_true,
    prmse_true,
    true_score_variance,
    variance_of_errors,
)

# allow test directory to be set via an environment variable
# which is needed for package testing
TEST_DIR = os.environ.get("TESTDIR", None)
if TEST_DIR:
    rsmtool_test_dir = TEST_DIR
else:
    from rsmtool.test_utils import rsmtool_test_dir


class TestUtilsPrmse(unittest.TestCase):
    def test_compute_n_human_scores(self):
        df = pd.DataFrame(
            {"h1": [1, 2, 3, 4], "h2": [1, None, 2, None], "h3": [None, None, 1, None]}
        )
        expected_n = pd.Series([2, 1, 3, 1])
        n_scores = get_n_human_scores(df)
        assert_array_equal(expected_n, n_scores)

    def test_compute_n_human_scores_zeros(self):
        df = pd.DataFrame(
            {"h1": [1, 2, 3, None], "h2": [1, None, 2, None], "h3": [None, None, 1, None]}
        )
        expected_n = pd.Series([2, 1, 3, 0])
        n_scores = get_n_human_scores(df)
        assert_array_equal(expected_n, n_scores)

    def test_compute_n_human_scores_array(self):
        df = pd.DataFrame(
            {"h1": [1, 2, 3, None], "h2": [1, None, 2, None], "h3": [None, None, 1, None]}
        )
        arr = df.to_numpy()
        expected_n = pd.Series([2, 1, 3, 0])
        n_scores = get_n_human_scores(arr)
        assert_array_equal(expected_n, n_scores)

    def test_prmse_single_human_ve(self):
        df = pd.DataFrame({"system": [1, 2, 5], "sc1": [2, 3, 5]})
        prmse = prmse_true(df["system"], df["sc1"], 0.5)
        self.assertEqual(prmse, 0.9090909090909091)

    def test_prmse_single_human_ve_array_as_input(self):
        system_scores = np.array([1, 2, 5])
        human_scores = np.array([2, 3, 5])
        prmse = prmse_true(system_scores, human_scores, 0.5)
        self.assertEqual(prmse, 0.9090909090909091)

    def test_variance_of_errors_all_single_scored(self):
        # this test should raise a UserWarning
        sc1 = [1, 2, 3, None, None]
        sc2 = [None, None, None, 2, 3]
        df = pd.DataFrame({"sc1": sc1, "sc2": sc2})
        with warnings.catch_warnings(record=True) as warning_list:
            variance_of_errors_human = variance_of_errors(df)
        self.assertTrue(variance_of_errors_human is None)
        assert issubclass(warning_list[-1].category, UserWarning)

    def test_prmse_all_single_scored(self):
        # this test should raise a UserWarning
        system_scores = [1, 2, 3, 4, 5]
        sc1 = [1, 2, 3, None, None]
        sc2 = [None, None, None, 2, 3]
        df = pd.DataFrame({"sc1": sc1, "sc2": sc2, "system": system_scores})
        with warnings.catch_warnings(record=True) as warning_list:
            prmse = prmse_true(df["system"], df[["sc1", "sc2"]])
        self.assertTrue(prmse is None)
        assert issubclass(warning_list[-1].category, UserWarning)

    def test_get_true_score_evaluations_single_human_no_ve(self):
        df = pd.DataFrame({"system": [1, 2, 5], "sc1": [2, 3, 5]})
        with self.assertRaises(ValueError):
            get_true_score_evaluations(df, "system", "sc1")


class TestPrmseJohnsonData(unittest.TestCase):
    """
    This class tests the PRMSE functions against the benchmarks
    provided by Matt Johnson who did the original derivation and
    implemented the function in R. This test ensures that Python
    implementation results in the same values
    """

    human_score_columns = ["h1", "h2", "h3", "h4"]
    system_score_columns = ["system"]

    @classmethod
    def setUpClass(cls):
        full_matrix_file = Path(rsmtool_test_dir) / "data" / "files" / "prmse_data.csv"
        sparse_matrix_file = (
            Path(rsmtool_test_dir) / "data" / "files" / "prmse_data_sparse_matrix.csv"
        )
        cls.data_full = pd.read_csv(full_matrix_file)
        cls.data_sparse = pd.read_csv(sparse_matrix_file)

    def test_variance_of_errors_full_matrix(self):
        human_scores = self.human_score_columns
        df_humans = self.data_full[human_scores]
        variance_errors_human = variance_of_errors(df_humans)
        expected_v_e = 0.509375
        self.assertEqual(variance_errors_human, expected_v_e)

    def test_variance_of_errors_sparse_matrix(self):
        human_scores = self.human_score_columns
        df_humans = self.data_sparse[human_scores]
        variance_errors_human = variance_of_errors(df_humans)
        expected_v_e = 0.5150882
        self.assertAlmostEqual(variance_errors_human, expected_v_e, 7)

    def test_variance_of_true_scores_full_matrix(self):
        human_scores = self.human_score_columns
        df_humans = self.data_full[human_scores]
        variance_errors_human = 0.509375
        expected_var_true = 0.7765515
        var_true = true_score_variance(df_humans, variance_errors_human)
        self.assertAlmostEqual(var_true, expected_var_true, 7)

    def test_variance_of_true_scores_sparse_matrix(self):
        human_scores = self.human_score_columns
        df_humans = self.data_sparse[human_scores]
        variance_errors_human = 0.5150882
        expected_var_true = 0.769816
        var_true = true_score_variance(df_humans, variance_errors_human)
        self.assertAlmostEqual(var_true, expected_var_true, 7)

    def test_variance_of_true_scores_sparse_matrix_computed_ve(self):
        human_scores = self.human_score_columns
        df_humans = self.data_sparse[human_scores]
        expected_var_true = 0.769816
        var_true = true_score_variance(df_humans)
        self.assertAlmostEqual(var_true, expected_var_true, 7)

    def test_mse_full_matrix(self):
        human_scores = self.human_score_columns
        df_humans = self.data_full[human_scores]
        system = self.data_full["system"]
        variance_errors_human = 0.509375
        expected_mse_true = 0.3564625
        mse = mse_true(system, df_humans, variance_errors_human)
        self.assertAlmostEqual(mse, expected_mse_true, 7)

    def test_mse_sparse_matrix(self):
        human_scores = self.human_score_columns
        df_humans = self.data_sparse[human_scores]
        system = self.data_sparse["system"]
        variance_errors_human = 0.5150882
        expected_mse_true = 0.3550792
        mse = mse_true(system, df_humans, variance_errors_human)
        self.assertAlmostEqual(mse, expected_mse_true, 7)

    def test_mse_sparse_matrix_computed_ve(self):
        human_scores = self.human_score_columns
        df_humans = self.data_sparse[human_scores]
        system = self.data_sparse["system"]
        expected_mse_true = 0.3550792
        mse = mse_true(system, df_humans)
        self.assertAlmostEqual(mse, expected_mse_true, 7)

    def test_prmse_full_matrix_given_ve(self):
        human_scores = self.human_score_columns
        df_humans = self.data_full[human_scores]
        system = self.data_full["system"]
        variance_errors_human = 0.509375
        expected_prmse_true = 0.5409673
        prmse = prmse_true(system, df_humans, variance_errors_human)
        self.assertAlmostEqual(prmse, expected_prmse_true, 7)

    def test_prmse_sparse_matrix_given_ve(self):
        human_scores = self.human_score_columns
        df_humans = self.data_sparse[human_scores]
        system = self.data_sparse["system"]
        variance_errors_human = 0.5150882
        expected_prmse_true = 0.538748
        prmse = prmse_true(system, df_humans, variance_errors_human)
        self.assertAlmostEqual(prmse, expected_prmse_true, 7)

    def test_prmse_full_matrix_computed_ve(self):
        human_scores = self.human_score_columns
        df_humans = self.data_full[human_scores]
        system = self.data_full["system"]
        expected_prmse_true = 0.5409673
        prmse = prmse_true(system, df_humans)
        self.assertAlmostEqual(prmse, expected_prmse_true, 7)

    def test_prmse_sparse_matrix_computed_ve(self):
        human_scores = self.human_score_columns
        df_humans = self.data_sparse[human_scores]
        system = self.data_sparse["system"]
        expected_prmse_true = 0.538748
        prmse = prmse_true(system, df_humans)
        self.assertAlmostEqual(prmse, expected_prmse_true, 7)

    def test_prmse_sparse_matrix_array_as_input(self):
        human_scores = self.human_score_columns
        df_humans = self.data_sparse[human_scores].to_numpy()
        system = np.array(self.data_sparse["system"])
        expected_prmse_true = 0.538748
        prmse = prmse_true(system, df_humans)
        self.assertAlmostEqual(prmse, expected_prmse_true, 7)

    def test_compute_true_score_evaluations_full(self):
        expected_df = pd.DataFrame(
            {
                "N": 10000,
                "N raters": 4,
                "N single": 0,
                "N multiple": 10000,
                "Variance of errors": 0.509375,
                "True score var": 0.7765515,
                "MSE true": 0.3564625,
                "PRMSE true": 0.5409673,
            },
            index=["system"],
        )
        df_prmse = get_true_score_evaluations(
            self.data_full, self.system_score_columns, self.human_score_columns
        )
        assert_frame_equal(df_prmse, expected_df, check_dtype=False)

    def test_compute_true_score_evaluations_sparse(self):
        expected_df = pd.DataFrame(
            {
                "N": 10000,
                "N raters": 4,
                "N single": 3421,
                "N multiple": 6579,
                "Variance of errors": 0.5150882,
                "True score var": 0.769816,
                "MSE true": 0.3550792,
                "PRMSE true": 0.538748,
            },
            index=["system"],
        )
        df_prmse = get_true_score_evaluations(
            self.data_sparse, self.system_score_columns, self.human_score_columns
        )
        assert_frame_equal(df_prmse, expected_df, check_dtype=False)

    def test_compute_true_score_evaluations_given_ve(self):
        expected_df = pd.DataFrame(
            {
                "N": 10000,
                "N raters": 4,
                "N single": 3421,
                "N multiple": 6579,
                "Variance of errors": 0.5150882,
                "True score var": 0.769816,
                "MSE true": 0.3550792,
                "PRMSE true": 0.538748,
            },
            index=["system"],
        )
        df_prmse = get_true_score_evaluations(
            self.data_sparse,
            self.system_score_columns,
            self.human_score_columns,
            variance_errors_human=0.5150882,
        )
        assert_frame_equal(df_prmse, expected_df, check_dtype=False)
