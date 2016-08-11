"""
Unit tests for testing functions in comparison.py

:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
"""

import pandas as pd

from pandas.util.testing import assert_frame_equal


from rsmtool.comparison import process_confusion_matrix


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
