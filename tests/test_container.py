import unittest
import warnings

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from rsmtool.container import DataContainer


class TestBuilderDataContainer(unittest.TestCase):
    """Test class for BuilderDataContainer."""

    def test_copy(self):
        expected = pd.DataFrame(
            [
                ["John", 1, 5.0],
                ["Mary", 2, 4.0],
                ["Sally", 6, np.nan],
                ["Jeff", 3, 9.0],
                ["Edwin", 9, 1.0],
            ],
            columns=["string", "numeric", "numeric_missing"],
        )

        container = DataContainer([{"frame": expected, "name": "test", "path": "foo"}])
        new_container = container.copy()

        self.assertNotEqual(id(new_container), id(container))
        for name in new_container.keys():
            frame = new_container.get_frame(name)
            path = new_container.get_path(name)

            old_frame = container.get_frame(name)
            old_path = container.get_path(name)

            self.assertEqual(path, old_path)
            assert_frame_equal(frame, old_frame)
            self.assertNotEqual(id(frame), id(old_frame))

    def test_copy_not_deep(self):
        expected = pd.DataFrame(
            [
                ["John", 1, 5.0],
                ["Mary", 2, 4.0],
                ["Sally", 6, np.nan],
                ["Jeff", 3, 9.0],
                ["Edwin", 9, 1.0],
            ],
            columns=["string", "numeric", "numeric_missing"],
        )

        container = DataContainer([{"frame": expected, "name": "test", "path": "foo"}])
        new_container = container.copy(deep=False)

        self.assertNotEqual(id(new_container), id(container))
        for name in new_container.keys():
            frame = new_container.get_frame(name)
            path = new_container.get_path(name)

            old_frame = container.get_frame(name)
            old_path = container.get_path(name)

            self.assertEqual(path, old_path)
            assert_frame_equal(frame, old_frame)
            self.assertEqual(id(frame), id(old_frame))

    def test_rename(self):
        expected = pd.DataFrame(
            [
                ["John", 1, 5.0],
                ["Mary", 2, 4.0],
                ["Sally", 6, np.nan],
                ["Jeff", 3, 9.0],
                ["Edwin", 9, 1.0],
            ],
            columns=["string", "numeric", "numeric_missing"],
        )

        container = DataContainer([{"frame": expected, "name": "test"}])
        container.rename("test", "flerf")
        assert_frame_equal(container.flerf, expected)

    def test_rename_with_path(self):
        expected = pd.DataFrame(
            [
                ["John", 1, 5.0],
                ["Mary", 2, 4.0],
                ["Sally", 6, np.nan],
                ["Jeff", 3, 9.0],
                ["Edwin", 9, 1.0],
            ],
            columns=["string", "numeric", "numeric_missing"],
        )

        container = DataContainer([{"frame": expected, "name": "test", "path": "foo"}])
        container.rename("test", "flerf")
        self.assertEqual(container.get_path("flerf"), "foo")

    def test_drop(self):
        container = DataContainer([{"frame": pd.DataFrame(), "name": "test"}])
        container.drop("test")
        self.assertFalse("test" in container)

    def test_drop_warning(self):
        container = DataContainer([{"frame": pd.DataFrame(), "name": "test"}])

        with self.assertRaises(Warning):
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                container.drop("flab")

    def test_get_frames_by_prefix(self):
        container = DataContainer(
            [
                {"frame": pd.DataFrame(), "name": "test_two"},
                {"frame": pd.DataFrame(), "name": "test_three"},
                {"frame": pd.DataFrame(), "name": "exclude"},
            ]
        )

        frames = container.get_frames(prefix="test")
        self.assertEqual(sorted(list(frames.keys())), sorted(["test_two", "test_three"]))

    def test_get_frames_by_suffix(self):
        container = DataContainer(
            [
                {"frame": pd.DataFrame(), "name": "include_this_one"},
                {"frame": pd.DataFrame(), "name": "include_this_one_not"},
                {"frame": pd.DataFrame(), "name": "we_want_this_one"},
            ]
        )

        frames = container.get_frames(suffix="one")
        self.assertEqual(
            sorted(list(frames.keys())),
            sorted(["include_this_one", "we_want_this_one"]),
        )

    def test_get_frames_both_suffix_and_prefix(self):
        container = DataContainer(
            [
                {"frame": pd.DataFrame(), "name": "include_this_frame"},
                {"frame": pd.DataFrame(), "name": "include_it"},
                {"frame": pd.DataFrame(), "name": "exclude_frame"},
                {"frame": pd.DataFrame(), "name": "include_this_other_frame"},
            ]
        )

        frames = container.get_frames(prefix="include", suffix="frame")
        self.assertEqual(
            sorted(list(frames.keys())),
            sorted(["include_this_frame", "include_this_other_frame"]),
        )

    def test_get_frames_no_match(self):
        container = DataContainer(
            [
                {"frame": pd.DataFrame(), "name": "include_this_one"},
                {"frame": pd.DataFrame(), "name": "include_this_one_not"},
                {"frame": pd.DataFrame(), "name": "we_want_this_one"},
            ]
        )

        frames = container.get_frames(suffix="foo")
        self.assertEqual(frames, {})
