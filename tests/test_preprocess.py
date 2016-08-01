import random
import sys
import warnings

from os.path import dirname, join, normpath

import numpy as np
import pandas as pd

from nose.tools import assert_raises, eq_, ok_, raises
from numpy.testing import assert_array_equal, assert_almost_equal
from pandas.util.testing import assert_frame_equal

_my_dir = dirname(__file__)
_code_dir = normpath(join(_my_dir, '..', 'code'))
sys.path.append(_code_dir)

from rsmtool.preprocess import (apply_inverse_transform,
                                apply_sqrt_transform,
                                apply_log_transform,
                                apply_add_one_log_transform,
                                apply_add_one_inverse_transform,
                                filter_on_column,
                                filter_on_flag_columns,
                                remove_outliers,
                                transform_feature)


def test_filter_on_column():
    data = {'spkitemlab': np.arange(1, 9, dtype='int64'), 'sc1': ['00', 'TD', '02', '03'] * 2}
    bad_df = pd.DataFrame(data=data)

    df_filtered_with_zeros = pd.DataFrame({'spkitemlab': [1, 3, 4, 5, 7, 8], 'sc1': [0.0, 2.0, 3.0] * 2})
    df_filtered = pd.DataFrame({'spkitemlab': [3, 4, 7, 8], 'sc1': [2.0, 3.0] * 2})

    output_df_with_zeros, output_excluded_df_with_zeros = filter_on_column(bad_df, 'sc1', 'spkitemlab', exclude_zeros=False)
    output_df, output_excluded_df = filter_on_column(bad_df, 'sc1', 'spkitemlab', exclude_zeros=True)
    assert_frame_equal(output_df_with_zeros, df_filtered_with_zeros)
    assert_frame_equal(output_df, df_filtered)


def test_filter_on_column_all_non_numeric():
    data = {'spkitemlab': range(1, 9), 'sc1': ['A', 'I', 'TD', 'TD'] * 2}
    bad_df = pd.DataFrame(data=data)

    expected_df_excluded = bad_df.copy()
    expected_df_excluded['sc1'] = np.nan

    df_filtered, df_excluded = filter_on_column(bad_df, 'sc1', 'spkitemlab', exclude_zeros=False)
    ok_(df_filtered.empty)
    assert_frame_equal(df_excluded, expected_df_excluded)


def test_filter_on_column_std_epsilon_zero():
    # Test that the function exclude columns where std is returned as
    # very low value rather than 0
    data = {'id': np.arange(1, 21, dtype='int64'),
            'feature_zero_sd': [1.5601] * 20,
            'feature_ok': np.arange(1, 21)}
    bad_df = pd.DataFrame(data=data)

    output_df, output_excluded_df = filter_on_column(bad_df,
                                                     'feature_zero_sd',
                                                     'id',
                                                     exclude_zeros=False,
                                                     exclude_zero_sd=True)

    good_df = bad_df[['feature_ok', 'id']].copy()
    assert_frame_equal(output_df, good_df)
    ok_(output_excluded_df.empty)


def test_transform_feature():
    name = 'dpsec'
    data = np.array([1, 2, 3, 4])
    # run the test but suppress the expected runtime warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        assert_raises(ValueError, transform_feature, name, data, 'add_one_inverse')
        assert_array_equal(transform_feature(name, data, 'inv'), 1/data)
        assert_array_equal(transform_feature(name, data, 'raw'), data)
        assert_array_equal(transform_feature(name, data, 'org'), data)
        assert_array_equal(transform_feature(name, data, 'log'), np.log(data))
        assert_array_equal(transform_feature(name, data, 'addOneInv'), 1/(data+1))
        assert_array_equal(transform_feature(name, data, 'addOneLn'), np.log(data+1))


def test_transform_feature_with_warning():
    name = 'dpsec'
    data = np.array([-1, 0, 2, 3])
    # run the test but suppress the expected runtime warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        assert_array_equal(transform_feature(name, data, 'sqrt', raise_error=False),
                           np.sqrt(data))


def test_transform_feature_with_error():
    name = 'dpsec'
    data = np.array([-1, 0, 2, 3])
    # run the test but suppress the expected runtime warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        assert_raises(ValueError, transform_feature, name, data, 'sqrt')


def test_apply_inverse_transform():
    assert_raises(ValueError, apply_inverse_transform, 'name', np.array([0, 1, 2]))
    assert_raises(ValueError, apply_inverse_transform, 'name', np.array([-2, -3, 1, 2]))
    assert_array_equal(apply_inverse_transform('name', np.array([0, 2, 4]), raise_error=False),
                       np.array([np.inf, 0.5, 0.25]))
    assert_array_equal(apply_inverse_transform('name', np.array([-2, -4, 1]), raise_error=False),
                       np.array([-0.5, -0.25, 1]))
    assert_array_equal(apply_inverse_transform('name', np.array([2, 4])), np.array([0.5, 0.25]))
    assert_array_equal(apply_inverse_transform('name', np.array([-2, -4])), np.array([-0.5, -0.25]))


def test_apply_sqrt_transform():
    assert_raises(ValueError, apply_sqrt_transform, 'name', np.array([-2, -3, 1, 2]))
    assert_array_equal(apply_sqrt_transform('name', np.array([-1, 2, 4]), raise_error=False),
                       np.array([np.nan, np.sqrt(2), 2]))
    assert_array_equal(apply_sqrt_transform('name', np.array([2, 4])), np.array([np.sqrt(2), 2]))
    assert_array_equal(apply_sqrt_transform('name', np.array([0.5, 4])), np.array([np.sqrt(0.5), 2]))
    assert_array_equal(apply_sqrt_transform('name', np.array([0, 4])), np.array([0, 2]))


def test_apply_log_transform():
    assert_raises(ValueError, apply_log_transform, 'name', np.array([-1, 2, 3]))
    assert_raises(ValueError, apply_log_transform, 'name', np.array([0, 2, 3]))
    assert_array_equal(apply_log_transform('name', np.array([-1, 1, 4]), raise_error=False),
                       np.array([np.nan, np.log(1), np.log(4)]))
    assert_array_equal(apply_log_transform('name', np.array([0, 1, 4]), raise_error=False),
                       np.array([-np.inf, np.log(1), np.log(4)]))
    assert_array_equal(apply_log_transform('name', np.array([1, 4])), np.array([np.log(1), np.log(4)]))


def test_apply_add_one_inverse_transform():
    assert_raises(ValueError, apply_add_one_inverse_transform, 'name', np.array([-1, -2, 3, 5]))
    assert_array_equal(apply_add_one_inverse_transform('name', np.array([-2, -3, 1, 4]), raise_error=False),
                       np.array([-1, -1/2, 1/2, 1/5]))
    assert_array_equal(apply_add_one_inverse_transform('name', np.array([1, 4])), np.array([1/2, 1/5]))
    assert_array_equal(apply_add_one_inverse_transform('name', np.array([0, 4])), np.array([1, 1/5]))


def test_apply_add_one_log_transform():
    assert_raises(ValueError, apply_add_one_log_transform, 'name', np.array([-2, -3, 2, 3]))
    assert_array_equal(apply_add_one_log_transform('name', np.array([-2, -0.5, 2, 4]), raise_error=False),
                       np.array([np.nan, np.log(0.5), np.log(3), np.log(5)]))
    assert_array_equal(apply_add_one_log_transform('name', np.array([2, 4])), np.array([np.log(3), np.log(5)]))
    assert_array_equal(apply_add_one_log_transform('name', np.array([0, 4])), np.array([0, np.log(5)]))


def test_filter_on_flag_column_nothing_to_exclude():
    df = pd.DataFrame({'spkitemid': ['a', 'b', 'c', 'd'],
                       'sc1': [1, 2, 1, 3],
                       'feature': [2, 3, 4, 5],
                       'flag1': [0, 0, 0, 0],
                       'flag2': [1, 2, 2, 1]})
    flag_dict = {'flag1': [0], 'flag2': [1, 2]}
    df_new, df_excluded = filter_on_flag_columns(df, flag_dict)
    assert_frame_equal(df_new, df)
    eq_(len(df_excluded), 0)


def test_filter_on_flag_column_empty_flag_dictionary():
    df = pd.DataFrame({'spkitemid': ['a', 'b', 'c', 'd'],
                       'sc1': [1, 2, 1, 3],
                       'feature': [2, 3, 4, 5],
                       'flag1': [0, 0, 0, 0],
                       'flag2': [1, 2, 2, 1]})
    flag_dict = {}
    df_new, df_excluded = filter_on_flag_columns(df, flag_dict)
    assert_frame_equal(df_new, df)
    eq_(len(df_excluded), 0)


def test_filter_on_flag_column_nothing_to_exclude_convert_to_float():
    df = pd.DataFrame({'spkitemid': ['a', 'b', 'c', 'd'],
                       'sc1': [1, 2, 1, 3],
                       'feature': [2, 3, 4, 5],
                       'flag1': [0.0, 0.0, 0.0, 0.0],
                       'flag2': [1, 'TD', 'TD', 1]})

    df_new = pd.DataFrame({'spkitemid': ['a', 'b', 'c', 'd'],
                           'sc1': [1, 2, 1, 3],
                           'feature': [2, 3, 4, 5],
                           'flag1': [0.0, 0.0, 0.0, 0.0],
                           'flag2': [1.0, 'TD', 'TD', 1.0]})
    flag_dict = {'flag1': [0.0], 'flag2': [1.0, 'TD']}
    df_filtered, df_excluded = filter_on_flag_columns(df, flag_dict)
    print(df_filtered, df_excluded)
    assert_frame_equal(df_filtered, df_new)
    eq_(len(df_excluded), 0)


def test_filter_on_flag_column():
    good_df = pd.DataFrame({'spkitemid': ['a', 'b', 'c', 'd'],
                            'sc1': [1, 2, 1, 3],
                            'feature': [2, 3, 4, 5],
                            'flag1': [0, 0, 0, 0],
                            'flag2': [1.0, 2.0, 3.0, 1.0]})
    bad_df = pd.DataFrame({'spkitemid': ['a1', 'b1', 'c1', 'd1'],
                           'sc1': [1, 2, 1, 3],
                           'feature': [2, 3, 4, 5],
                           'flag1': [1, 0, 20, 14],
                           'flag2': [1.0, 5.0, 'TD', None]})
    df = pd.concat([good_df, bad_df])

    flag_dict = {'flag1': [0], 'flag2': [1, 2, 3]}

    df_new, df_excluded = filter_on_flag_columns(df, flag_dict)
    assert_array_equal(df_new['spkitemid'], ['a', 'b', 'c', 'd'])
    assert_array_equal(df_excluded['spkitemid'], ['a1', 'b1', 'c1', 'd1'])


@raises(KeyError)
def test_filter_on_flag_column_missing_columns():
    df = pd.DataFrame({'spkitemid': ['a', 'b', 'c', 'd'],
                       'sc1': [1, 2, 1, 3],
                       'feature': [2, 3, 4, 5],
                       'flag1': ['1', '1', '1', '1'],
                       'flag2': ['1', '2', '2', '1']})
    flag_dict = {'flag3': ['0'], 'flag2': ['1', '2']}
    df_new, df_excluded = filter_on_flag_columns(df, flag_dict)


@raises(ValueError)
def test_filter_on_flag_column_nothing_left():
    bad_df = pd.DataFrame({'spkitemid': ['a1', 'b1', 'c1', 'd1'],
                           'sc1': [1, 2, 1, 3],
                           'feature': [2, 3, 4, 5],
                           'flag1': [1, 0, 20, 14],
                           'flag2': [1, 1.0, 'TD', '03']})

    flag_dict = {'flag1': ['0'], 'flag2': ['1', '2', '3']}

    df_new, df_excluded = filter_on_flag_columns(bad_df, flag_dict)


def test_remove_outliers():
    # we want to test that even if we pass in a list of
    # integers, we still get the right clamped output
    data = [1, 1, 2, 2, 1, 1]*10 + [10]
    ceiling = np.mean(data) + 4*np.std(data)
    clamped_data = remove_outliers(data)
    assert_almost_equal(clamped_data[-1], ceiling)
