#python

# Test functions for prmse_utils.py

import numpy as np
import pandas as pd

from nose.tools import (assert_equal,
                        eq_,
                        raises,
                        assert_almost_equal)

from os.path import dirname

from rsmtool.prmse_utils import (compute_variance_of_errors,
                                 compute_true_score_var_subset_double_scored,
                                 compute_true_score_var_all_double_scored,
                                 compute_mse_subset_double_scored,
                                 compute_mse_all_double_scored)


# get the directory containing the tests
test_dir = dirname(__file__)

def test_compute_variance_of_erors_zero():
    sc1 = [1, 2, 3, 1, 2, 3]
    sc2 = [1, 2, 3, 1, 2, 3]
    df = pd.DataFrame({'sc1': sc1,
                       'sc2': sc2})
    eq_(compute_variance_of_errors(df), 0)



def test_compute_variance_of_erors_one():
    sc1 = [1, 2, 3, 1, 2]
    sc2 = [2, 1, 4, 2, 3]
    df = pd.DataFrame({'sc1': sc1,
                       'sc2': sc2})
    eq_(compute_variance_of_errors(df), 0.5)

@raises(ValueError)
def test_compute_variance_of_erors_error():
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
                                                    v_e), 4/3)


def test_compute_true_score_compare():
    single = pd.Series([])
    double = pd.Series([2, 1, 3, 2, 3, 4, 6])

    v_e = 0.3

    variance_subset = compute_true_score_var_subset_double_scored(single,                                                  double,
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
    eq_(mse, 2/3)


def test_compute_mse_compare():
    human_double = pd.Series([1, 1, 1, 1, 1, 1])
    system_double = pd.Series([2, 2, 0, 0, 1, 1])
    v_e = 0.3
    mse_subset = compute_mse_subset_double_scored(pd.Series([]),
                                                  human_double,
                                                  pd.Series([]),
                                                  system_double,
                                                  v_e)
    mse_all = compute_mse_all_double_scored(human_double,
                                            system_double,
                                            v_e)
    assert_almost_equal(mse_subset, mse_all)


