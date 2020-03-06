import numpy as np
import pandas as pd

from numpy.testing import assert_array_equal

from nose.tools import (eq_,
                        raises,
                        assert_almost_equal)

from pathlib import Path

from rsmtool.utils.prmse import (true_score_variance,
                                 variance_of_errors,
                                 mse_true,
                                 prmse_true,
                                 get_n_human_scores)


# get the directory containing the tests
test_dir = Path(__file__).parent


@raises(ValueError)
def test_variance_of_errors_all_single_scored():
    sc1 = [1, 2, 3, None, None]
    sc2 = [None, None, None, 2, 3]
    df = pd.DataFrame({'sc1': sc1,
                       'sc2': sc2})
    variance_of_errors(df)


def test_compute_n_human_scores():
    df = pd.DataFrame({'h1': [1, 2, 3, 4],
                       'h2': [1, None, 2, None],
                       'h3': [None, None, 1, None]})
    expected_n = pd.Series([2, 1, 3, 1])
    n_scores = get_n_human_scores(df)
    assert_array_equal(expected_n, n_scores)


def test_compute_n_human_scores_zeros():
    df = pd.DataFrame({'h1': [1, 2, 3, None],
                       'h2': [1, None, 2, None],
                       'h3': [None, None, 1, None]})
    expected_n = pd.Series([2, 1, 3, 0])
    n_scores = get_n_human_scores(df)
    assert_array_equal(expected_n, n_scores)



class TestPrmseJohnsonData():

    '''
    This class tests the PRMSE functions against the benchmarks
    provided by Matt Johnson who did the original derivation and
    implemented the function in R. This test ensures that Python
    implementation results in the same values
    '''

    def setUp(self):
        full_matrix_file = test_dir / 'data' / 'files' / 'prmse_data.csv'
        sparse_matrix_file = test_dir / 'data' / 'files' / 'prmse_data_sparse_matrix.csv'
        self.data_full = pd.read_csv(full_matrix_file)
        self.data_sparse = pd.read_csv(sparse_matrix_file)
        self.human_score_columns = ['h1', 'h2', 'h3', 'h4']
        self.system_score_columns = ['system']


    def test_variance_of_errors_full_matrix(self):
        human_scores = self.human_score_columns
        df_humans = self.data_full[human_scores]
        variance_errors_human = variance_of_errors(df_humans)
        expected_v_e = 0.509375
        eq_(variance_errors_human, expected_v_e)


    def test_variance_of_errors_sparse_matrix(self):
        human_scores = self.human_score_columns
        df_humans = self.data_sparse[human_scores]
        variance_errors_human= variance_of_errors(df_humans)
        expected_v_e = 0.5150882
        assert_almost_equal(variance_errors_human, expected_v_e, 7)

    def test_variance_of_true_scores_full_matrix(self):
        human_scores = self.human_score_columns
        df_humans = self.data_full[human_scores]
        variance_errors_human = 0.509375
        expected_var_true = 0.7765515
        var_true = true_score_variance(df_humans,
                                       variance_errors_human)
        assert_almost_equal(var_true, expected_var_true, 7)

    def test_variance_of_true_scores_sparse_matrix(self):
        human_scores = self.human_score_columns
        df_humans = self.data_sparse[human_scores]
        variance_errors_human = 0.5150882
        expected_var_true = 0.769816
        var_true = true_score_variance(df_humans,
                                       variance_errors_human)
        assert_almost_equal(var_true, expected_var_true, 7)

    def test_variance_of_true_scores_sparse_matrix_computed_ve(self):
        human_scores = self.human_score_columns
        df_humans = self.data_sparse[human_scores]
        expected_var_true = 0.769816
        var_true = true_score_variance(df_humans)
        assert_almost_equal(var_true, expected_var_true, 7)

    def test_mse_full_matrix(self):
        human_scores = self.human_score_columns
        df_humans = self.data_full[human_scores]
        system = self.data_full['system']
        variance_errors_human = 0.509375
        expected_mse_true = 0.3564625
        mse = mse_true(system,
                       df_humans,
                       variance_errors_human)
        assert_almost_equal(mse, expected_mse_true, 7)

    def test_mse_sparse_matrix(self):
        human_scores = self.human_score_columns
        df_humans = self.data_sparse[human_scores]
        system = self.data_sparse['system']
        variance_errors_human = 0.5150882
        expected_mse_true = 0.3550792
        mse = mse_true(system,
                       df_humans,
                       variance_errors_human)
        assert_almost_equal(mse, expected_mse_true, 7)


    def test_mse_sparse_matrix_computed_ve(self):
        human_scores = self.human_score_columns
        df_humans = self.data_sparse[human_scores]
        system = self.data_sparse['system']
        expected_mse_true = 0.3550792
        mse = mse_true(system,
                       df_humans)
        assert_almost_equal(mse, expected_mse_true, 7)

    def test_prmse_full_matrix_given_ve(self):
        human_scores = self.human_score_columns
        df_humans = self.data_full[human_scores]
        system = self.data_full['system']
        variance_errors_human = 0.509375
        expected_prmse_true = 0.5409673
        prmse = prmse_true(system,
                           df_humans,
                           variance_errors_human)
        assert_almost_equal(prmse, expected_prmse_true, 7)

    def test_prmse_sparse_matrix_given_ve(self):
        human_scores = self.human_score_columns
        df_humans = self.data_sparse[human_scores]
        system = self.data_sparse['system']
        variance_errors_human = 0.5150882
        expected_prmse_true = 0.538748
        prmse = prmse_true(system,
                           df_humans,
                           variance_errors_human)
        assert_almost_equal(prmse, expected_prmse_true, 7)


    def test_prmse_full_matrix_computed_ve(self):
        human_scores = self.human_score_columns
        df_humans = self.data_full[human_scores]
        system = self.data_full['system']
        expected_prmse_true = 0.5409673
        prmse = prmse_true(system,
                           df_humans)
        assert_almost_equal(prmse, expected_prmse_true, 7)

    def test_prmse_sparse_matrix_computed_ve(self):
        human_scores = self.human_score_columns
        df_humans = self.data_sparse[human_scores]
        system = self.data_sparse['system']
        expected_prmse_true = 0.538748
        prmse = prmse_true(system,
                           df_humans)
        assert_almost_equal(prmse, expected_prmse_true, 7)


    def test_prmse_sparse_matrix_array_as_input(self):
        human_scores = self.human_score_columns
        df_humans = self.data_sparse[human_scores].to_numpy()
        system = np.array(self.data_sparse['system'])
        expected_prmse_true = 0.538748
        prmse = prmse_true(system,
                           df_humans)
        assert_almost_equal(prmse, expected_prmse_true, 7)



