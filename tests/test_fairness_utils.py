"""
Tests for function in fairness_utils.py
"""


import numpy as np
import pandas as pd

from nose.tools import (eq_,
                        ok_,
                        raises,
                        assert_almost_equal)

from os.path import dirname

from rsmtool.fairness_utils import (convert_to_ordered_category,
                                    get_fairness_analyses)


# get the directory containing the tests
test_dir = dirname(__file__)



def test_convert_to_ordered_category():
    values = pd.Series(['a', 'a', 'b', 'b', 'b',
                         'c', 'c', 'd'])
    cat_values = convert_to_ordered_category(values)
    eq_(cat_values.cat.categories[0], 'b')


def test_convert_to_ordered_category_several_maximums():
    values = pd.Series(['a_2', 'a_3', 'a_1',
                       'a_2', 'a_3', 'a_1',
                       'a_2', 'a_3', 'a_1'])
    cat_values = convert_to_ordered_category(values)
    eq_(cat_values.cat.categories[0], 'a_1')

def test_convert_to_ordered_category_base_category():
    values = pd.Series(['a', 'a', 'b', 'b', 'b',
                         'c', 'c', 'd'])
    cat_values = convert_to_ordered_category(values, 'a')
    eq_(cat_values.cat.categories[0], 'a')

@raises(ValueError)
def test_convert_to_ordered_category_wrong_base_category():
    values = pd.Series(['a', 'a', 'b', 'b', 'b',
                         'c', 'c', 'd'])
    cat_values = convert_to_ordered_category(values, 'e')

def test_get_fairness_analyses_no_effect():
    df = pd.DataFrame({'sc1': [1, 1, 4, 4],
                       'raw': [0.9, 0.99, 3.99, 3.9],
                       'L1': ['Klingon',
                              'Esperanto',
                              'Klingon',
                              'Esperanto']})
    (model_dict,
     dc) = get_fairness_analyses(df,
                                'L1',
                                 'raw')
    eq_(len(model_dict), 3)
    eq_(len(dc), 4)
    sig_osa = dc['fairness_metrics_by_L1'].loc['sig',
                                               'Overall score accuracy']
    sig_osd = dc['fairness_metrics_by_L1'].loc['sig',
                                               'Overall score difference']
    sig_csd = dc['fairness_metrics_by_L1'].loc['sig',
                                               'Conditional score difference']
    eq_(sig_osa, 1)
    eq_(sig_osd, 1)
    eq_(sig_csd, 1)

def test_get_fairness_analyses_osa_difference():
    df = pd.DataFrame({'sc1': [1, 1, 4, 4],
                       'raw': [0.9, 5, 3.99, 0.01],
                       'L1': ['Klingon',
                              'Esperanto',
                              'Klingon',
                              'Esperanto']})
    (model_dict,
     dc) = get_fairness_analyses(df,
                                'L1',
                                 'raw')
    eq_(len(model_dict), 3)
    eq_(len(dc), 4)
    sig_osa = dc['fairness_metrics_by_L1'].loc['sig',
                                               'Overall score accuracy']
    assert_almost_equal(sig_osa, 0, places=4)


def test_get_fairness_analyses_osd_difference():
    df = pd.DataFrame({'sc1': [1, 1, 4, 4],
                       'raw': [0.49, 1.5, 3.5, 4.51],
                       'L1': ['Klingon',
                              'Esperanto',
                              'Klingon',
                              'Esperanto']})
    (model_dict,
     dc) = get_fairness_analyses(df,
                                'L1',
                                 'raw')
    eq_(len(model_dict), 3)
    eq_(len(dc), 4)
    sig_osa = dc['fairness_metrics_by_L1'].loc['sig',
                                               'Overall score accuracy']
    sig_osd = dc['fairness_metrics_by_L1'].loc['sig',
                                               'Overall score difference']
    eq_(sig_osa, 1)
    assert_almost_equal(sig_osd, 0, places=4)


def test_get_fairness_analyses_csd_difference():
    df = pd.DataFrame({'sc1': [1, 1, 1, 1, 4, 4, 4, 4],
                       'raw': [1.21, 1.2, 1.21, 1.32, 3.78, 3.91, 3.9, 3.9],
                       'L1': ['Klingon',
                              'Klingon',
                              'Klingon',
                              'Esperanto',
                              'Klingon',
                              'Esperanto',
                              'Esperanto',
                              'Esperanto']})
    (model_dict,
     dc) = get_fairness_analyses(df,
                                'L1',
                                 'raw')
    eq_(len(model_dict), 3)
    eq_(len(dc), 4)
    sig_osa = dc['fairness_metrics_by_L1'].loc['sig',
                                               'Overall score accuracy']
    sig_osd = dc['fairness_metrics_by_L1'].loc['sig',
                                               'Overall score difference']
    sig_csd = dc['fairness_metrics_by_L1'].loc['sig',
                                               'Conditional score difference']
    ok_(sig_osa > 0.5)
    ok_(sig_osd > 0.5)
    assert_almost_equal(sig_csd, 0, places=4)






