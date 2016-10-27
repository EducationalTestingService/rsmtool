import sys

import numpy as np
from numpy.testing import assert_array_equal

import pandas as pd

from nose.tools import assert_equal, assert_raises

from rsmtool.create_features import (select_by_prefix,
                                     select_by_subset,
                                     generate_default_specs,
                                     generate_specs_from_data)


def test_select_by_prefix():
    fnames = ['1gram\thtis', '1gram\tis', '1gram\ta', '1gram\thouse', '2gram\this_is', '2gram\tis_a', '2gram\ta_house', '3gram\tthis_is_a', '3gram\tis_a_house']
    assert_raises(ValueError, select_by_prefix, fnames, ['5gram'])
    assert_array_equal(select_by_prefix(fnames, ['1gram']), ['1gram\thtis', '1gram\tis', '1gram\ta', '1gram\thouse'])
    assert_array_equal(select_by_prefix(fnames, ['1gram', '3gram']), ['1gram\thtis', '1gram\tis', '1gram\ta', '1gram\thouse', '3gram\tthis_is_a', '3gram\tis_a_house'])


def test_select_by_subset():
    feature_subset_specs = pd.DataFrame({'Feature': ['Grammar',
                                                     'Vocabulary',
                                                     'Fluency',
                                                     'Content_coverage',
                                                     'Discourse',
                                                     'Pronunciation',
                                                     'Prosody',
                                                     'Content_accuracy'],
                                        'high_entropy': [1, 1, 1, 1, 1, 1, 1, 0],
                                        'low_entropy': [0, 0, 1, 0, 0, 1, 1, 1]})
    # This list should also trigger a warning about extra subset features not in the data
    fnames = ['Grammar', 'Vocabulary', 'Pronunciation', 'Content_accuracy']
    high_entropy_fnames = ['Grammar', 'Vocabulary', 'Pronunciation']
    assert_array_equal(select_by_subset(fnames, feature_subset_specs, 'high_entropy'), high_entropy_fnames)


def test_select_by_subset_warnings():
    feature_subset_specs = pd.DataFrame({'Feature': ['Grammar',
                                                     'Vocabulary',
                                                     'Fluency',
                                                     'Content_coverage',
                                                     'Discourse',
                                                     'Pronunciation',
                                                     'Prosody',
                                                     'Content_accuracy'],
                                        'high_entropy': [1, 1, 1, 1, 1, 1, 1, 0],
                                        'low_entropy': [0, 0, 1, 0, 0, 1, 1, 1]})
    extra_fnames = ['Grammar', 'Vocabulary', 'Rhythm']
    assert_array_equal(select_by_subset(extra_fnames, feature_subset_specs, 'high_entropy'), ['Grammar', 'Vocabulary'])


def test_generate_default_specs():
    fnames = ['Grammar', 'Vocabulary', 'Pronunciation']
    df_specs = generate_default_specs(fnames)
    assert_equal(len(df_specs), 3)
    assert_equal(df_specs['feature'][0], 'Grammar')
    assert_equal(df_specs['transform'][1], 'raw')
    assert_equal(df_specs['sign'][2], 1.0)


def test_generate_specs_from_data_with_negative_sign():
    feature_subset_specs = pd.DataFrame({'Feature': ['Grammar',
                                                     'Vocabulary',
                                                     'Fluency',
                                                     'Content_coverage',
                                                     'Discourse'],
                                        'Sign_SYS1': ['-', '+', '+', '+', '-']})
    np.random.seed(10)
    data = {'Grammar': np.random.randn(10), 'Fluency': np.random.randn(10), 'Discourse': np.random.randn(10), 'r1': np.random.choice(4, 10), 'spkitemlab': ['a-5'] * 10}
    df = pd.DataFrame(data)
    df_specs = generate_specs_from_data(['Grammar', 'Fluency', 'Discourse'],
                                        'r1',
                                        df,
                                        feature_subset_specs,
                                        'SYS1')
    assert_equal(len(df_specs), 3)
    assert_array_equal(df_specs['feature'], ['Grammar', 'Fluency', 'Discourse'])
    assert_array_equal(df_specs['sign'], [-1.0, 1.0, -1.0])


def test_generate_specs_from_data_with_default_sign():
    feature_subset_specs = pd.DataFrame({'Feature': ['Grammar',
                                                     'Vocabulary',
                                                     'Fluency',
                                                     'Content_coverage',
                                                     'Discourse'],
                                        'Sign_SYS1': ['-', '+', '+', '+', '-']})
    np.random.seed(10)
    data = {'Grammar': np.random.randn(10), 'Fluency': np.random.randn(10), 'Discourse': np.random.randn(10), 'r1': np.random.choice(4, 10), 'spkitemlab': ['a-5'] * 10}
    df = pd.DataFrame(data)
    df = pd.DataFrame(data)
    df_specs = generate_specs_from_data(['Grammar', 'Fluency', 'Discourse'],
                                        'r1',
                                        df,
                                        feature_subset_specs,
                                        feature_sign=None)
    assert_equal(len(df_specs), 3)
    assert_array_equal(df_specs['feature'], ['Grammar', 'Fluency', 'Discourse'])
    assert_array_equal(df_specs['sign'], [1.0, 1.0, 1.0])


def test_generate_specs_from_data_with_transformation():
    feature_subset_specs = pd.DataFrame({'Feature': ['Grammar',
                                                     'Vocabulary',
                                                     'Fluency',
                                                     'Content_coverage',
                                                     'Discourse'],
                                        'Sign_SYS1': ['-', '+', '+', '+', '-']})
    np.random.seed(10)
    r1 = np.random.choice(range(1, 5), 10)
    data = {'Grammar': np.random.randn(10), 'Vocabulary': r1**2, 'Discourse': np.random.randn(10), 'r1': r1, 'spkitemlab': ['a-5'] * 10}
    df = pd.DataFrame(data)
    df_specs = generate_specs_from_data(['Grammar', 'Vocabulary', 'Discourse'],
                                        'r1',
                                        df,
                                        feature_subset_specs,
                                        'SYS1')
    assert_array_equal(df_specs['feature'], ['Grammar', 'Vocabulary', 'Discourse'])
    assert_equal(df_specs['transform'][1], 'sqrt')

def test_generate_specs_from_data_when_transformation_changes_sign():
    feature_subset_specs = pd.DataFrame({'Feature': ['Grammar',
                                                     'Vocabulary',
                                                     'Fluency',
                                                     'Content_coverage',
                                                     'Discourse'],
                                        'Sign_SYS1': ['-', '+', '+', '+', '-']})
    np.random.seed(10)
    r1 = np.random.choice(range(1, 5), 10)
    data = {'Grammar': np.random.randn(10), 'Vocabulary': 1/r1, 'Discourse': np.random.randn(10), 'r1': r1, 'spkitemlab': ['a-5'] * 10}
    df = pd.DataFrame(data)
    df_specs = generate_specs_from_data(['Grammar', 'Vocabulary', 'Discourse'],
                                        'r1',
                                        df,
                                        feature_subset_specs,
                                        'SYS1')
    assert_equal(df_specs['feature'][1], 'Vocabulary')
    assert_equal(df_specs['transform'][1], 'addOneInv')
    assert_equal(df_specs['sign'][1], -1)


def test_generate_specs_from_data_no_subset_specs():
    np.random.seed(10)
    data = {'Grammar': np.random.randn(10), 'Fluency': np.random.randn(10), 'Discourse': np.random.randn(10), 'r1': np.random.choice(4, 10), 'spkitemlab': ['a-5'] * 10}
    df = pd.DataFrame(data)
    df = pd.DataFrame(data)
    df_specs = generate_specs_from_data(['Grammar', 'Fluency', 'Discourse'],
                                        'r1',
                                        df)
    assert_equal(len(df_specs), 3)
    assert_array_equal(df_specs['feature'], ['Grammar', 'Fluency', 'Discourse'])
    assert_array_equal(df_specs['sign'], [1.0, 1.0, 1.0])


