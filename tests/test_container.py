"""
"""

import numpy as np
import pandas as pd

from nose.tools import assert_false, eq_
from pandas.util.testing import assert_frame_equal
from rsmtool.container import DataContainer


class TestBuilderDataContainer:

    def test_rename(self):

        expected = pd.DataFrame([['John', 1, 5.0],
                                 ['Mary', 2, 4.0],
                                 ['Sally', 6, np.nan],
                                 ['Jeff', 3, 9.0],
                                 ['Edwin', 9, 1.0]],
                                columns=['string', 'numeric', 'numeric_missing'])

        container = DataContainer([{'frame': expected, 'name': 'test'}])
        container.rename('test', 'flerf')
        assert_frame_equal(container.flerf, expected)

    def test_drop(self):

        container = DataContainer([{'frame': pd.DataFrame(), 'name': 'test'}])
        container.drop('test')
        assert_false('test' in container)

    def test_get_frames_by_prefix(self):

        container = DataContainer([{'frame': pd.DataFrame(), 'name': 'test_two'},
                                   {'frame': pd.DataFrame(), 'name': 'test_three'},
                                   {'frame': pd.DataFrame(), 'name': 'exclude'}])

        frames = container.get_frames(prefix='test')
        eq_(sorted(list(frames.keys())), sorted(['test_two', 'test_three']))

    def test_get_frames_by_suffix(self):

        container = DataContainer([{'frame': pd.DataFrame(), 'name': 'include_this_one'},
                                   {'frame': pd.DataFrame(), 'name': 'include_this_one_not'},
                                   {'frame': pd.DataFrame(), 'name': 'we_want_this_one'}])

        frames = container.get_frames(prefix='one')
        eq_(sorted(list(frames.keys())), sorted(['include_this_one', 'we_want_this_one']))
