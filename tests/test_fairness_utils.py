"""
Tests for function in fairness_utils.py
"""


import numpy as np
import pandas as pd

from nose.tools import (eq_,
                        raises,
                        assert_almost_equal)

from os.path import dirname

from rsmtool.fairness_utils import convert_to_ordered_category


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




