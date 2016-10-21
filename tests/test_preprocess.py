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


def test_filter_on_column_with_inf():
    # Test that the function exclude columns where feature value is 'inf'
    data = pd.DataFrame({'id': np.arange(1, 5, dtype='int64'),
                        'feature_1': [1.5601, 0, 2.33, 11.32],
                        'feature_ok': np.arange(1, 5)})
    data['feature_with_inf'] = 1/data['feature_1']
    bad_df = data[np.isinf(data['feature_with_inf'])].copy()
    good_df = data[~np.isinf(data['feature_with_inf'])].copy()
    bad_df.reset_index(drop=True, inplace=True)
    good_df.reset_index(drop=True, inplace=True)

    output_df, output_excluded_df = filter_on_column(data,
                                                     'feature_with_inf',
                                                     'id',
                                                     exclude_zeros=False,
                                                     exclude_zero_sd=True)

    print(output_df)
    assert_frame_equal(output_df, good_df)
    assert_frame_equal(output_excluded_df, bad_df)



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
        assert_array_equal(transform_feature(name, data, 'inv', raise_error=False),
                           1/data)
        assert_array_equal(transform_feature(name, data, 'addOneInv', raise_error=False),
                           1/(data+1))        
        assert_array_equal(transform_feature(name, data, 'log', raise_error=False),
                           np.log(data))        
        assert_array_equal(transform_feature(name, data, 'addOneLn', raise_error=False),
                           np.log(data+1))

def test_transform_feature_with_error():
    name = 'dpsec'
    data = np.array([-1, 0, 2, 3])
    # run the test but suppress the expected runtime warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        assert_raises(ValueError, transform_feature, name, data, 'sqrt')
        assert_raises(ValueError, transform_feature, name, data, 'inv')
        assert_raises(ValueError, transform_feature, name, data, 'addOneInv')
        assert_raises(ValueError, transform_feature, name, data, 'log')
        assert_raises(ValueError, transform_feature, name, data, 'addOneLn')



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


def test_filter_on_flag_column_empty_flag_dictionary():
    # no flags specified, keep the data frame as is
    df = pd.DataFrame({'spkitemid': ['a', 'b', 'c', 'd'],
                       'sc1': [1, 2, 1, 3],
                       'feature': [2, 3, 4, 5],
                       'flag1': [0, 0, 0, 0],
                       'flag2': [1, 2, 2, 1]})
    flag_dict = {}
    df_new, df_excluded = filter_on_flag_columns(df, flag_dict)
    assert_frame_equal(df_new, df)
    eq_(len(df_excluded), 0)



def test_filter_on_flag_column_nothing_to_exclude_int_column_and_dict():
    df = pd.DataFrame({'spkitemid': ['a', 'b', 'c', 'd'],
                       'sc1': [1, 2, 1, 3],
                       'feature': [2, 3, 4, 5],
                       'flag1': [0, 1, 2, 3]})
    flag_dict = {'flag1': [0, 1, 2, 3, 4]}
    df_new, df_excluded = filter_on_flag_columns(df, flag_dict)
    assert_frame_equal(df_new, df)
    eq_(len(df_excluded), 0)


def test_filter_on_flag_column_nothing_to_exclude_float_column_and_dict():
    df = pd.DataFrame({'spkitemid': ['a', 'b', 'c', 'd'],
                       'sc1': [1, 2, 1, 3],
                       'feature': [2, 3, 4, 5],
                       'flag1': [0.5, 1.1, 2.2, 3.6]})
    flag_dict = {'flag1': [0.5, 1.1, 2.2, 3.6, 4.5]}
    df_new, df_excluded = filter_on_flag_columns(df, flag_dict)
    assert_frame_equal(df_new, df)
    eq_(len(df_excluded), 0)

def test_filter_on_flag_column_nothing_to_exclude_str_column_and_dict():
    df = pd.DataFrame({'spkitemid': ['a', 'b', 'c', 'd'],
                       'sc1': [1, 2, 1, 3],
                       'feature': [2, 3, 4, 5],
                       'flag1': ['a', 'b', 'c', 'd']})
    flag_dict = {'flag1': ['a', 'b', 'c', 'd', 'e']}
    df_new, df_excluded = filter_on_flag_columns(df, flag_dict)
    assert_frame_equal(df_new, df)
    eq_(len(df_excluded), 0)


def test_filter_on_flag_column_nothing_to_exclude_float_column_int_dict():
    df = pd.DataFrame({'spkitemid': ['a', 'b', 'c', 'd'],
                       'sc1': [1, 2, 1, 3],
                       'feature': [2, 3, 4, 5],
                       'flag1': [0.0, 1.0, 2.0, 3.0]})
    flag_dict = {'flag1': [0, 1, 2, 3, 4]}
    df_new, df_excluded = filter_on_flag_columns(df, flag_dict)
    assert_frame_equal(df_new, df)
    eq_(len(df_excluded), 0)


def test_filter_on_flag_column_nothing_to_exclude_int_column_float_dict():
    df = pd.DataFrame({'spkitemid': ['a', 'b', 'c', 'd'],
                       'sc1': [1, 2, 1, 3],
                       'feature': [2, 3, 4, 5],
                       'flag1': [0, 1, 2, 3]})
    flag_dict = {'flag1': [0.0, 1.0, 2.0, 3.0, 4.5]}
    df_new, df_excluded = filter_on_flag_columns(df, flag_dict)
    assert_frame_equal(df_new, df)
    eq_(len(df_excluded), 0)


def test_filter_on_flag_column_nothing_to_exclude_str_column_float_dict():
    df = pd.DataFrame({'spkitemid': ['a', 'b', 'c', 'd'],
                       'sc1': [1, 2, 1, 3],
                       'feature': [2, 3, 4, 5],
                       'flag1': ['4', '1', '2', '3.5']})
    flag_dict = {'flag1': [0.0, 1.0, 2.0, 3.5, 4.0]}
    df_new, df_excluded = filter_on_flag_columns(df, flag_dict)
    assert_frame_equal(df_new, df)
    eq_(len(df_excluded), 0)


def test_filter_on_flag_column_nothing_to_exclude_float_column_str_dict():
    df = pd.DataFrame({'spkitemid': ['a', 'b', 'c', 'd'],
                       'sc1': [1, 2, 1, 3],
                       'feature': [2, 3, 4, 5],
                       'flag1': [4.0, 1.0, 2.0, 3.5]})
    flag_dict = {'flag1': ['1', '2', '3.5', '4', 'TD']}
    df_new, df_excluded = filter_on_flag_columns(df, flag_dict)
    assert_frame_equal(df_new, df)
    eq_(len(df_excluded), 0)


def test_filter_on_flag_column_nothing_to_exclude_str_column_int_dict():
    df = pd.DataFrame({'spkitemid': ['a', 'b', 'c', 'd'],
                       'sc1': [1, 2, 1, 3],
                       'feature': [2, 3, 4, 5],
                       'flag1': ['0.0', '1.0', '2.0', '3.0']})
    flag_dict = {'flag1': [0, 1, 2, 3, 4]}
    df_new, df_excluded = filter_on_flag_columns(df, flag_dict)
    assert_frame_equal(df_new, df)
    eq_(len(df_excluded), 0)


def test_filter_on_flag_column_nothing_to_exclude_int_column_str_dict():
    df = pd.DataFrame({'spkitemid': ['a', 'b', 'c', 'd'],
                       'sc1': [1, 2, 1, 3],
                       'feature': [2, 3, 4, 5],
                       'flag1': [0, 1, 2, 3]})
    flag_dict = {'flag1': ['0.0', '1.0', '2.0', '3.0', 'TD']}
    df_new, df_excluded = filter_on_flag_columns(df, flag_dict)
    assert_frame_equal(df_new, df)
    eq_(len(df_excluded), 0)


def test_filter_on_flag_column_nothing_to_exclude_mixed_type_column_str_dict():
    df = pd.DataFrame({'spkitemid': ['a', 'b', 'c', 'd'],
                       'sc1': [1, 2, 1, 3],
                       'feature': [2, 3, 4, 5],
                       'flag1': [0, '1.0', 2, 3.5]})
    flag_dict = {'flag1': ['0.0', '1.0', '2.0', '3.5', 'TD']}
    df_new, df_excluded = filter_on_flag_columns(df, flag_dict)
    print(df_new, df)
    assert_frame_equal(df_new, df)
    eq_(len(df_excluded), 0)


def test_filter_on_flag_column_nothing_to_exclude_mixed_type_column_int_dict():
    df = pd.DataFrame({'spkitemid': ['a', 'b', 'c', 'd'],
                       'sc1': [1, 2, 1, 3],
                       'feature': [2, 3, 4, 5],
                       'flag1': [0, '1.0', 2, 3.0]})
    flag_dict = {'flag1': [0, 1, 2, 3, 4]}
    df_new, df_excluded = filter_on_flag_columns(df, flag_dict)
    assert_frame_equal(df_new, df)
    eq_(len(df_excluded), 0)


def test_filter_on_flag_column_nothing_to_exclude_mixed_type_column_float_dict():
    df = pd.DataFrame({'spkitemid': ['a', 'b', 'c', 'd'],
                       'sc1': [1, 2, 1, 3],
                       'feature': [2, 3, 4, 5],
                       'flag1': [0, '1.5', 2, 3.5]})
    flag_dict = {'flag1': [0.0, 1.5, 2.0, 3.5, 4.0]}
    df_new, df_excluded = filter_on_flag_columns(df, flag_dict)
    assert_frame_equal(df_new, df)
    eq_(len(df_excluded), 0)


def test_filter_on_flag_column_nothing_to_exclude_int_column_mixed_type_dict():
    df = pd.DataFrame({'spkitemid': ['a', 'b', 'c', 'd'],
                       'sc1': [1, 2, 1, 3],
                       'feature': [2, 3, 4, 5],
                       'flag1': [0, 1, 2, 3]})
    flag_dict = {'flag1': [0, 1, 2, 3.0, 3.5, 'TD']}
    df_new, df_excluded = filter_on_flag_columns(df, flag_dict)
    assert_frame_equal(df_new, df)
    eq_(len(df_excluded), 0)


def test_filter_on_flag_column_nothing_to_exclude_float_column_mixed_type_dict():
    df = pd.DataFrame({'spkitemid': ['a', 'b', 'c', 'd'],
                       'sc1': [1, 2, 1, 3],
                       'feature': [2, 3, 4, 5],
                       'flag1': [0.0, 1.0, 2.0, 3.5]})
    flag_dict = {'flag1': [0, 1, 2, 3.0, 3.5, 'TD']}
    df_new, df_excluded = filter_on_flag_columns(df, flag_dict)
    assert_frame_equal(df_new, df)
    eq_(len(df_excluded), 0)


def test_filter_on_flag_column_nothing_to_exclude_str_column_mixed_type_dict():
    df = pd.DataFrame({'spkitemid': ['a', 'b', 'c', 'd'],
                       'sc1': [1, 2, 1, 3],
                       'feature': [2, 3, 4, 5],
                       'flag1': ['0.0', '1.0', '2.0', '3.5']})
    flag_dict = {'flag1': [0, 1, 2, 3.0, 3.5, 'TD']}
    df_new, df_excluded = filter_on_flag_columns(df, flag_dict)
    assert_frame_equal(df_new, df)
    eq_(len(df_excluded), 0)


def test_filter_on_flag_column_nothing_to_exclude_mixed_type_column_mixed_type_dict():
    df = pd.DataFrame({'spkitemid': ['a', 'b', 'c', 'd'],
                       'sc1': [1, 2, 1, 3],
                       'feature': [2, 3, 4, 5],
                       'flag1': [1, 2, 3.5, 'TD']})
    flag_dict = {'flag1': [0, 1, 2, 3.0, 3.5, 'TD']}
    df_new, df_excluded = filter_on_flag_columns(df, flag_dict)
    assert_frame_equal(df_new, df)
    eq_(len(df_excluded), 0)


def test_filter_on_flag_column_mixed_type_column_mixed_type_dict_filter_preserve_type():
    df = pd.DataFrame({'spkitemid': ['a', 'b', 'c', 'd', 'e', 'f'],
                       'sc1': [1, 2, 1, 3, 4, 5],
                       'feature': [2, 3, 4, 5, 6, 2],
                       'flag1': [1, 1.5, 2, 3.5, 'TD', 'NS']})
    flag_dict = {'flag1': [1.5, 2, 'TD']}

    df_new_expected = pd.DataFrame({'spkitemid': ['b', 'c', 'e'],
                                    'sc1': [2, 1, 4],
                                    'feature': [3, 4, 6],
                                    'flag1': [1.5, 2,'TD']})

    df_excluded_expected = pd.DataFrame({'spkitemid': ['a', 'd', 'f'],
                                         'sc1': [1, 3, 5],
                                         'feature': [2, 5, 2],
                                         'flag1': [1, 3.5, 'NS']})

    df_new, df_excluded = filter_on_flag_columns(df, flag_dict)
    assert_frame_equal(df_new, df_new_expected)
    assert_frame_equal(df_excluded, df_excluded_expected)


def test_filter_on_flag_column_with_none_value_in_int_flag_column_int_dict():
    df = pd.DataFrame({'spkitemid': [1, 2, 3, 4, 5, 6],
                       'sc1': [1, 2, 1, 3, 4, 5],
                       'feature': [2, 3, 4, 5, 6, 2],
                       'flag1': [1, 2, 2, 3, 4, None]}, 
                       dtype=object)
    flag_dict = {'flag1': [2, 4]}

    df_new_expected = pd.DataFrame({'spkitemid': [2, 3, 5],
                                    'sc1': [2, 1, 4],
                                    'feature': [3, 4, 6],
                                    'flag1': [2, 2, 4]},
                                    dtype=object)

    df_excluded_expected = pd.DataFrame({'spkitemid': [1, 4, 6],
                                         'sc1': [1, 3, 5],
                                         'feature': [2, 5, 2],
                                         'flag1': [1, 3, None]}, 
                                         dtype=object)

    df_new, df_excluded = filter_on_flag_columns(df, flag_dict)
    assert_frame_equal(df_new, df_new_expected)
    assert_frame_equal(df_excluded, df_excluded_expected)



def test_filter_on_flag_column_with_none_value_in_float_flag_column_float_dict():
    df = pd.DataFrame({'spkitemid': ['a', 'b', 'c', 'd', 'e', 'f'],
                       'sc1': [1, 2, 1, 3, 4, 5],
                       'feature': [2, 3, 4, 5, 6, 2],
                       'flag1': [1.2, 2.1, 2.1, 3.3, 4.2, None]})
    flag_dict = {'flag1': [2.1, 4.2]}

    df_new_expected = pd.DataFrame({'spkitemid': ['b', 'c', 'e'],
                                    'sc1': [2, 1, 4],
                                    'feature': [3, 4, 6],
                                    'flag1': [2.1, 2.1, 4.2]})

    df_excluded_expected = pd.DataFrame({'spkitemid': ['a', 'd', 'f'],
                                         'sc1': [1, 3, 5],
                                         'feature': [2, 5, 2],
                                         'flag1': [1.2, 3.3, None]})

    df_new, df_excluded = filter_on_flag_columns(df, flag_dict)
    assert_frame_equal(df_new, df_new_expected)
    assert_frame_equal(df_excluded, df_excluded_expected)


def test_filter_on_flag_column_with_none_value_in_str_flag_column_str_dict():
    df = pd.DataFrame({'spkitemid': ['a', 'b', 'c', 'd', 'e', 'f'],
                       'sc1': [1, 2, 1, 3, 4, 5],
                       'feature': [2, 3, 4, 5, 6, 2],
                       'flag1': ['a', 'b', 'b', 'c', 'd', None]})
    flag_dict = {'flag1': ['b', 'd']}

    df_new_expected = pd.DataFrame({'spkitemid': ['b', 'c', 'e'],
                                    'sc1': [2, 1, 4],
                                    'feature': [3, 4, 6],
                                    'flag1': ['b', 'b', 'd']})

    df_excluded_expected = pd.DataFrame({'spkitemid': ['a', 'd', 'f'],
                                         'sc1': [1, 3, 5],
                                         'feature': [2, 5, 2],
                                         'flag1': ['a', 'c', None]})

    df_new, df_excluded = filter_on_flag_columns(df, flag_dict)
    assert_frame_equal(df_new, df_new_expected)
    assert_frame_equal(df_excluded, df_excluded_expected)


def test_filter_on_flag_column_with_none_value_in_mixed_type_flag_column_float_dict():
    df = pd.DataFrame({'spkitemid': ['a', 'b', 'c', 'd', 'e', 'f'],
                       'sc1': [1, 2, 1, 3, 4, 5],
                       'feature': [2, 3, 4, 5, 6, 2],
                       'flag1': [1, 1.5, 2.0, 'TD', 2.0, None]})
    flag_dict = {'flag1': [1.5, 2.0]}

    df_new_expected = pd.DataFrame({'spkitemid': ['b', 'c', 'e'],
                                    'sc1': [2, 1, 4],
                                    'feature': [3, 4, 6],
                                    'flag1': [1.5, 2.0, 2.0]})

    df_excluded_expected = pd.DataFrame({'spkitemid': ['a', 'd', 'f'],
                                         'sc1': [1, 3, 5],
                                         'feature': [2, 5, 2],
                                         'flag1': [1, 'TD', None]})

    df_new, df_excluded = filter_on_flag_columns(df, flag_dict)
    assert_frame_equal(df_new, df_new_expected)
    assert_frame_equal(df_excluded, df_excluded_expected)



def test_filter_on_flag_column_with_none_value_in_mixed_type_flag_column_int_dict():
    df = pd.DataFrame({'spkitemid': ['a', 'b', 'c', 'd', 'e', 'f'],
                       'sc1': [1, 2, 1, 3, 4, 5],
                       'feature': [2, 3, 4, 5, 6, 2],
                       'flag1': [1.5, 2, 2, 'TD', 4, None]})
    flag_dict = {'flag1': [2, 4]}

    df_new_expected = pd.DataFrame({'spkitemid': ['b', 'c', 'e'],
                                    'sc1': [2, 1, 4],
                                    'feature': [3, 4, 6],
                                    'flag1': [2, 2, 4]})

    df_excluded_expected = pd.DataFrame({'spkitemid': ['a', 'd', 'f'],
                                         'sc1': [1, 3, 5],
                                         'feature': [2, 5, 2],
                                         'flag1': [1.5, 'TD', None]})

    df_new, df_excluded = filter_on_flag_columns(df, flag_dict)
    assert_frame_equal(df_new, df_new_expected)
    assert_frame_equal(df_excluded, df_excluded_expected)



def test_filter_on_flag_column_with_none_value_in_mixed_type_flag_column_mixed_type_dict():
    df = pd.DataFrame({'spkitemid': ['a', 'b', 'c', 'd', 'e', 'f'],
                       'sc1': [1, 2, 1, 3, 4, 5],
                       'feature': [2, 3, 4, 5, 6, 2],
                       'flag1': [1, 1.5, 2, 3.5, 'TD', None]})
    flag_dict = {'flag1': [1.5, 2, 'TD']}

    df_new_expected = pd.DataFrame({'spkitemid': ['b', 'c', 'e'],
                                    'sc1': [2, 1, 4],
                                    'feature': [3, 4, 6],
                                    'flag1': [1.5, 2,'TD']})

    df_excluded_expected = pd.DataFrame({'spkitemid': ['a', 'd', 'f'],
                                         'sc1': [1, 3, 5],
                                         'feature': [2, 5, 2],
                                         'flag1': [1, 3.5, 'NS']})

    df_new, df_excluded = filter_on_flag_columns(df, flag_dict)
    assert_frame_equal(df_new, df_new_expected)
    assert_frame_equal(df_excluded, df_excluded_expected)



def test_filter_on_flag_column_two_flags_same_responses():
    df = pd.DataFrame({'spkitemid': ['a', 'b', 'c', 'd', 'e', 'f'],
                       'sc1': [1, 2, 1, 3, 4, 5],
                       'feature': [2, 3, 4, 5, 6, 2],
                       'flag1': [1, 1.5, 2, 3.5, 'TD', 'NS'],
                       'flag2': [1, 0, 0, 1, 0, 1]})
    flag_dict = {'flag1': [1.5, 2, 'TD'], 'flag2': [0]}

    df_new_expected = pd.DataFrame({'spkitemid': ['b', 'c', 'e'],
                                    'sc1': [2, 1, 4],
                                    'feature': [3, 4, 6],
                                    'flag1': [1.5, 2,'TD'],
                                    'flag2': [0, 0, 0]})

    df_excluded_expected = pd.DataFrame({'spkitemid': ['a', 'd', 'f'],
                                         'sc1': [1, 3, 5],
                                         'feature': [2, 5, 2],
                                         'flag1': [1, 3.5, 'NS'],
                                         'flag2': [1, 1, 1]})

    df_new, df_excluded = filter_on_flag_columns(df, flag_dict)
    assert_frame_equal(df_new, df_new_expected)
    assert_frame_equal(df_excluded, df_excluded_expected)



def test_filter_on_flag_column_two_flags_different_responses():
    df = pd.DataFrame({'spkitemid': ['a', 'b', 'c', 'd', 'e', 'f'],
                       'sc1': [1, 2, 1, 3, 4, 5],
                       'feature': [2, 3, 4, 5, 6, 2],
                       'flag1': [1, 1.5, 2, 3.5, 'TD', 'NS'],
                       'flag2': [2, 0, 0, 1, 0, 1]})
    flag_dict = {'flag1': [1.5, 2, 'TD', 'NS'], 'flag2': [0, 2]}

    df_new_expected = pd.DataFrame({'spkitemid': ['b', 'c', 'e'],
                                    'sc1': [2, 1, 4],
                                    'feature': [3, 4, 6],
                                    'flag1': [1.5, 2,'TD'],
                                    'flag2': [2, 0, 0]})

    df_excluded_expected = pd.DataFrame({'spkitemid': ['a', 'd', 'f'],
                                         'sc1': [1, 3, 5],
                                         'feature': [2, 5, 2],
                                         'flag1': [0, 3.5, 'NS'],
                                         'flag2': [2, 1, 1]})

    df_new, df_excluded = filter_on_flag_columns(df, flag_dict)
    assert_frame_equal(df_new, df_new_expected)
    assert_frame_equal(df_excluded, df_excluded_expected)



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
