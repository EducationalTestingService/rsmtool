import numpy as np
import pandas as pd

from numpy.testing import assert_array_equal

from nose.tools import (eq_,
                        raises,
                        assert_almost_equal)

from pathlib import Path

from rsmtool.prmse_utils import (compute_variance_of_errors,
                                 compute_true_score_var_subset_double_scored,
                                 compute_true_score_var_all_double_scored,
                                 compute_mse_subset_double_scored,
                                 compute_mse_all_double_scored,
                                 compute_prmse)


# get the directory containing the tests
test_dir = Path(__file__).parent


def test_compute_variance_of_errors_zero():
    sc1 = [1, 2, 3, 1, 2, 3]
    sc2 = [1, 2, 3, 1, 2, 3]
    df = pd.DataFrame({'sc1': sc1,
                       'sc2': sc2})
    eq_(compute_variance_of_errors(df), 0)


def test_compute_variance_of_errors_one():
    sc1 = [1, 2, 3, 1, 2]
    sc2 = [2, 1, 4, 2, 3]
    df = pd.DataFrame({'sc1': sc1,
                       'sc2': sc2})
    eq_(compute_variance_of_errors(df), 0.5)


@raises(ValueError)
def test_compute_variance_of_errors_error():
    sc1 = [1, 2, 3, 1, None]
    sc2 = [2, 1, None, 2, 3]
    df = pd.DataFrame({'sc1': sc1,
                       'sc2': sc2})
    compute_variance_of_errors(df)


def test_compute_true_score_var_subset_zero():
    single = pd.Series([1, 1, 1, 1, 1, 1])
    double = pd.Series([1, 1, 1, 1, 1, 1])

    v_e = 0
    eq_(compute_true_score_var_subset_double_scored(single,
                                                    double,
                                                    v_e), 0)


def test_compute_true_score_var_subset():
    single = pd.Series([1, 2, 4, 1, 2, 2])
    double = pd.Series([4, 1, 1])

    v_e = 0.5
    eq_(compute_true_score_var_subset_double_scored(single,
                                                    double,
                                                    v_e), 4 / 3)


def test_compute_true_score_compare():
    single = pd.Series([], dtype='float64')
    double = pd.Series([2, 1, 3, 2, 3, 4, 6])

    v_e = 0.3

    variance_subset = compute_true_score_var_subset_double_scored(single,
                                                                  double,
                                                                  v_e)
    variance_all = compute_true_score_var_all_double_scored(double, v_e)

    eq_(variance_subset, variance_all)


def test_compute_mse_subset_double_scored_zero():
    human_single = pd.Series([1, 1, 1, 1, 1, 1])
    system_single = pd.Series([1, 1, 1, 1, 1, 1])
    v_e = 0
    mse = compute_mse_subset_double_scored(human_single,
                                           human_single,
                                           system_single,
                                           system_single,
                                           v_e)
    eq_(mse, 0)


def test_compute_mse_subset_double_scored():
    human_single = pd.Series([1, 1, 1, 1, 1, 1])
    system_single = pd.Series([2, 2, 0, 0, 1, 1])
    v_e = 0
    mse = compute_mse_subset_double_scored(human_single,
                                           human_single,
                                           system_single,
                                           system_single,
                                           v_e)
    eq_(mse, 2 / 3)


def test_compute_mse_compare():
    human_double = pd.Series([1, 1, 1, 1, 1, 1])
    system_double = pd.Series([2, 2, 0, 0, 1, 1])
    v_e = 0.3
    mse_subset = compute_mse_subset_double_scored(pd.Series([], dtype='float64'),
                                                  human_double,
                                                  pd.Series([], dtype='float64'),
                                                  system_double,
                                                  v_e)
    mse_all = compute_mse_all_double_scored(human_double,
                                            system_double,
                                            v_e)
    assert_almost_equal(mse_subset, mse_all)


def test_compute_prmse_zero():
    sc1 = [1, 2, 1, 3, 1, 2]
    sc2 = [np.nan, 1, np.nan, 3, np.nan, 3]
    system_correct = pd.Series([1, 1.5, 1, 3, 1, 2.5])
    system_wrong = pd.Series([1, 1, 4, 5, 3, 1])
    df = pd.DataFrame({'sc1': sc1,
                       'sc2': sc2,
                       'system_correct': system_correct,
                       'system_wrong': system_wrong})

    prmse = compute_prmse(df,
                          ['system_correct', 'system_wrong'])
    print(prmse)
    eq_(prmse.loc['system_correct', 'N'], 6)
    eq_(prmse.loc['system_correct', 'N_single'], 3)
    eq_(prmse.loc['system_correct', 'N_double'], 3)

