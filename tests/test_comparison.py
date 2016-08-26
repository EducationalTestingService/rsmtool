"""
Unit tests for testing functions in comparison.py

:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
"""

import pandas as pd

from nose.tools import assert_equal, assert_raises, eq_, ok_, raises
from pandas.util.testing import assert_frame_equal
from scipy.stats import pearsonr


from rsmtool.comparison import (process_confusion_matrix,
                                compute_correlations_between_versions)


def test_process_confusion_matrix():
    in_cm = pd.DataFrame({1: [2, 3, 5],
                          2: [2, 5, 7],
                          3: [1, 3, 6]},
                         index=[1, 2, 3])

    expected_out_cm = pd.DataFrame({'human 1': [2, 3, 5],
                                    'human 2': [2, 5, 7],
                                    'human 3': [1, 3, 6]},
                                   index=['machine 1', 'machine 2', 'machine 3'])

    out_cm = process_confusion_matrix(in_cm)
    assert_frame_equal(out_cm, expected_out_cm)


def test_process_confusion_matrix_with_zero():
    in_cm = pd.DataFrame({0: [2, 3, 5],
                          1: [2, 5, 7],
                          2: [1, 3, 6]},
                         index=[0, 1, 2])

    expected_out_cm = pd.DataFrame({'human 0': [2, 3, 5],
                                    'human 1': [2, 5, 7],
                                    'human 2': [1, 3, 6]},
                                   index=['machine 0', 'machine 1', 'machine 2'])

    out_cm = process_confusion_matrix(in_cm)
    assert_frame_equal(out_cm, expected_out_cm)


def test_compute_correlations_between_versions_default_columns():
    df_old = pd.DataFrame({'spkitemid': ['a', 'b', 'c'],
                           'feature1': [1.3, 1.5, 2.1],
                           'feature2': [1.1, 6.2, 2.1],
                           'sc1': [2, 3, 4]})
    df_new = pd.DataFrame({'spkitemid': ['a', 'b', 'c'],
                           'feature1': [-1.3, -1.5, -2.1],
                           'feature2': [1.1, 6.2, 2.1],
                           'sc1': [2, 3, 4]})
    df_cors = compute_correlations_between_versions(df_old, df_new)
    assert_equal(df_cors.get_value('feature1', 'old_new'), -1.0)
    assert_equal(df_cors.get_value('feature2', 'old_new'), 1.0)
    assert_equal(df_cors.get_value('feature1', 'human_old'), pearsonr(df_old['feature1'],
                                                                      df_old['sc1'])[0])
    assert_equal(df_cors.get_value('feature1', 'human_new'), pearsonr(df_new['feature1'],
                                                                      df_new['sc1'])[0])
    assert_equal(df_cors.get_value('feature1', "N"), 3)


def test_compute_correlations_between_versions_custom_columns():
    df_old = pd.DataFrame({'id': ['a', 'b', 'c'],
                           'feature1': [1.3, 1.5, 2.1],
                           'feature2': [1.1, 6.2, 2.1],
                           'r1': [2, 3, 4]})
    df_new = pd.DataFrame({'id': ['a', 'b', 'c'],
                           'feature1': [-1.3, -1.5, -2.1],
                           'feature2': [1.1, 6.2, 2.1],
                           'r1': [2, 3, 4]})
    df_cors = compute_correlations_between_versions(df_old, df_new,
                                                    human_score='r1', id_column='id')
    assert_equal(df_cors.get_value('feature1', 'old_new'), -1.0)
    assert_equal(df_cors.get_value('feature2', 'old_new'), 1.0)
    assert_equal(df_cors.get_value('feature1', 'human_old'), pearsonr(df_old['feature1'],
                                                                      df_old['r1'])[0])
    assert_equal(df_cors.get_value('feature1', 'human_new'), pearsonr(df_new['feature1'],
                                                                      df_new['r1'])[0])
    assert_equal(df_cors.get_value('feature1', "N"), 3)


@raises(ValueError)
def test_compute_correlations_between_versions_no_matching_feature():
    df_old = pd.DataFrame({'id': ['a', 'b', 'c'],
                           'feature1': [1.3, 1.5, 2.1],
                           'feature2': [1.1, 6.2, 2.1],
                           'r1': [2, 3, 4]})
    df_new = pd.DataFrame({'id': ['a', 'b', 'c'],
                           'feature3': [-1.3, -1.5, -2.1],
                           'feature4': [1.1, 6.2, 2.1],
                           'r1': [2, 3, 4]})
    df_cors = compute_correlations_between_versions(df_old, df_new,
                                                    human_score='r1', id_column='id')


@raises(ValueError)
def test_compute_correlations_between_versions_no_matching_ids():
    df_old = pd.DataFrame({'id': ['a', 'b', 'c'],
                           'feature1': [1.3, 1.5, 2.1],
                           'feature2': [1.1, 6.2, 2.1],
                           'r1': [2, 3, 4]})
    df_new = pd.DataFrame({'id': ['a1', 'b1', 'c1'],
                           'feature1': [-1.3, -1.5, -2.1],
                           'feature2': [1.1, 6.2, 2.1],
                           'r1': [2, 3, 4]})
    df_cors = compute_correlations_between_versions(df_old, df_new,
                                                    human_score='r1', id_column='id')
