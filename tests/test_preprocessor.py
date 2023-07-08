import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from pandas.testing import assert_frame_equal

from rsmtool.preprocessor import (
    FeaturePreprocessor,
    FeatureSpecsProcessor,
    FeatureSubsetProcessor,
)


class TestFeaturePreprocessor(unittest.TestCase):
    """Tests class for FeaturePreprocessor tests."""

    @classmethod
    def setUpClass(cls):
        cls.fpp = FeaturePreprocessor()

    def test_select_candidates_with_N_or_more_items(self):
        data = pd.DataFrame({"candidate": ["a"] * 3 + ["b"] * 2 + ["c"], "sc1": [2, 3, 1, 5, 6, 1]})
        df_included_expected = pd.DataFrame(
            {"candidate": ["a"] * 3 + ["b"] * 2, "sc1": [2, 3, 1, 5, 6]}
        )
        df_excluded_expected = pd.DataFrame({"candidate": ["c"], "sc1": [1]})

        (df_included, df_excluded) = self.fpp.select_candidates(data, 2)
        assert_frame_equal(df_included, df_included_expected)
        assert_frame_equal(df_excluded, df_excluded_expected)

    def test_select_candidates_with_N_or_more_items_all_included(self):
        data = pd.DataFrame(
            {"candidate": ["a"] * 2 + ["b"] * 2 + ["c"] * 2, "sc1": [2, 3, 1, 5, 6, 1]}
        )

        (df_included, df_excluded) = self.fpp.select_candidates(data, 2)
        assert_frame_equal(df_included, data)
        self.assertEqual(len(df_excluded), 0)

    def test_select_candidates_with_N_or_more_items_all_excluded(self):
        data = pd.DataFrame({"candidate": ["a"] * 3 + ["b"] * 2 + ["c"], "sc1": [2, 3, 1, 5, 6, 1]})

        (df_included, df_excluded) = self.fpp.select_candidates(data, 4)
        assert_frame_equal(df_excluded, data)
        self.assertEqual(len(df_included), 0)

    def test_select_candidates_with_N_or_more_items_custom_name(self):
        data = pd.DataFrame({"ID": ["a"] * 3 + ["b"] * 2 + ["c"], "sc1": [2, 3, 1, 5, 6, 1]})
        df_included_expected = pd.DataFrame({"ID": ["a"] * 3 + ["b"] * 2, "sc1": [2, 3, 1, 5, 6]})
        df_excluded_expected = pd.DataFrame({"ID": ["c"], "sc1": [1]})

        (df_included, df_excluded) = self.fpp.select_candidates(data, 2, "ID")
        assert_frame_equal(df_included, df_included_expected)
        assert_frame_equal(df_excluded, df_excluded_expected)

    def test_rename_no_columns(self):
        df = pd.DataFrame(
            columns=[
                "spkitemid",
                "sc1",
                "sc2",
                "length",
                "raw",
                "candidate",
                "feature1",
                "feature2",
            ]
        )

        df = self.fpp.rename_default_columns(
            df, [], "spkitemid", "sc1", "sc2", "length", "raw", "candidate"
        )
        assert_array_equal(
            df.columns,
            [
                "spkitemid",
                "sc1",
                "sc2",
                "length",
                "raw",
                "candidate",
                "feature1",
                "feature2",
            ],
        )

    def test_rename_no_columns_some_values_none(self):
        df = pd.DataFrame(columns=["spkitemid", "sc1", "sc2", "feature1", "feature2"])

        df = self.fpp.rename_default_columns(df, [], "spkitemid", "sc1", "sc2", None, None, None)
        assert_array_equal(df.columns, ["spkitemid", "sc1", "sc2", "feature1", "feature2"])

    def test_rename_no_used_columns_but_unused_columns_with_default_names(self):
        df = pd.DataFrame(columns=["spkitemid", "sc1", "sc2", "length", "feature1", "feature2"])

        df = self.fpp.rename_default_columns(df, [], "spkitemid", "sc1", "sc2", None, None, None)
        assert_array_equal(
            df.columns,
            ["spkitemid", "sc1", "sc2", "##length##", "feature1", "feature2"],
        )

    def test_rename_used_columns(self):
        df = pd.DataFrame(columns=["id", "r1", "r2", "words", "SR", "feature1", "feature2"])

        df = self.fpp.rename_default_columns(df, [], "id", "r1", "r2", "words", "SR", None)
        assert_array_equal(
            df.columns,
            ["spkitemid", "sc1", "sc2", "length", "raw", "feature1", "feature2"],
        )

    def test_rename_used_columns_and_unused_columns_with_default_names(self):
        df = pd.DataFrame(columns=["id", "r1", "r2", "words", "raw", "feature1", "feature2"])

        df = self.fpp.rename_default_columns(df, [], "id", "r1", "r2", "words", None, None)
        assert_array_equal(
            df.columns,
            ["spkitemid", "sc1", "sc2", "length", "##raw##", "feature1", "feature2"],
        )

    def test_rename_used_columns_with_swapped_names(self):
        df = pd.DataFrame(columns=["id", "sc1", "sc2", "raw", "words", "feature1", "feature2"])

        df = self.fpp.rename_default_columns(df, [], "id", "sc2", "sc1", "words", None, None)
        assert_array_equal(
            df.columns,
            ["spkitemid", "sc2", "sc1", "##raw##", "length", "feature1", "feature2"],
        )

    def test_rename_used_columns_but_not_features(self):
        df = pd.DataFrame(columns=["id", "sc1", "sc2", "length", "feature2"])

        df = self.fpp.rename_default_columns(df, ["length"], "id", "sc1", "sc2", None, None, None)
        assert_array_equal(df.columns, ["spkitemid", "sc1", "sc2", "length", "feature2"])

    def test_rename_candidate_column(self):
        df = pd.DataFrame(
            columns=[
                "spkitemid",
                "sc1",
                "sc2",
                "length",
                "apptNo",
                "feature1",
                "feature2",
            ]
        )

        df = self.fpp.rename_default_columns(
            df, [], "spkitemid", "sc1", "sc2", None, None, "apptNo"
        )
        assert_array_equal(
            df.columns,
            [
                "spkitemid",
                "sc1",
                "sc2",
                "##length##",
                "candidate",
                "feature1",
                "feature2",
            ],
        )

    def test_rename_candidate_named_sc2(self):
        df = pd.DataFrame(columns=["id", "sc1", "sc2", "question", "l1", "score"])
        df_renamed = self.fpp.rename_default_columns(
            df, [], "id", "sc1", None, None, "score", "sc2"
        )
        assert_array_equal(
            df_renamed.columns,
            ["spkitemid", "sc1", "candidate", "question", "l1", "raw"],
        )

    def test_check_subgroups_missing_columns(self):
        df = pd.DataFrame(columns=["a", "b", "c"])

        subgroups = ["a", "d"]

        with self.assertRaises(KeyError):
            self.fpp.check_subgroups(df, subgroups)

    def test_check_subgroups_nothing_to_replace(self):
        df = pd.DataFrame({"a": ["1", "2"], "b": ["32", "34"], "d": ["abc", "def"]})

        subgroups = ["a", "d"]
        df_out = self.fpp.check_subgroups(df, subgroups)
        assert_frame_equal(df_out, df)

    def test_check_subgroups_replace_empty(self):
        df = pd.DataFrame({"a": ["1", ""], "b": ["   ", "34"], "d": ["ab c", "   "]})

        subgroups = ["a", "d"]
        df_expected = pd.DataFrame(
            {"a": ["1", "No info"], "b": ["   ", "34"], "d": ["ab c", "No info"]}
        )
        df_out = self.fpp.check_subgroups(df, subgroups)
        assert_frame_equal(df_out, df_expected)

    def test_filter_on_column(self):
        bad_df = pd.DataFrame(
            {
                "spkitemlab": np.arange(1, 9, dtype="int64"),
                "sc1": ["00", "TD", "02", "03"] * 2,
            }
        )

        df_filtered_with_zeros = pd.DataFrame(
            {"spkitemlab": [1, 3, 4, 5, 7, 8], "sc1": [0.0, 2.0, 3.0] * 2}
        )
        df_filtered = pd.DataFrame({"spkitemlab": [3, 4, 7, 8], "sc1": [2.0, 3.0] * 2})

        (
            output_df_with_zeros,
            output_excluded_df_with_zeros,
        ) = self.fpp.filter_on_column(bad_df, "sc1", "spkitemlab", exclude_zeros=False)
        output_df, output_excluded_df = self.fpp.filter_on_column(
            bad_df, "sc1", "spkitemlab", exclude_zeros=True
        )
        assert_frame_equal(output_df_with_zeros, df_filtered_with_zeros)
        assert_frame_equal(output_df, df_filtered)

    def test_filter_on_column_all_non_numeric(self):
        bad_df = pd.DataFrame({"sc1": ["A", "I", "TD", "TD"] * 2, "spkitemlab": range(1, 9)})

        expected_df_excluded = bad_df.copy()
        expected_df_excluded.drop("sc1", axis=1, inplace=True)

        df_filtered, df_excluded = self.fpp.filter_on_column(
            bad_df, "sc1", "spkitemlab", exclude_zeros=True
        )

        self.assertTrue(df_filtered.empty)
        self.assertTrue("sc1" not in df_filtered.columns)
        assert_frame_equal(df_excluded, expected_df_excluded, check_dtype=False)

    def test_filter_on_column_std_epsilon_zero(self):
        # Test that the function exclude columns where std is returned as
        # very low value rather than 0
        data = {
            "id": np.arange(1, 21, dtype="int64"),
            "feature_ok": np.arange(1, 21),
            "feature_zero_sd": [1.5601] * 20,
        }
        bad_df = pd.DataFrame(data=data)

        output_df, output_excluded_df = self.fpp.filter_on_column(
            bad_df, "feature_zero_sd", "id", exclude_zeros=False, exclude_zero_sd=True
        )

        good_df = bad_df[["id", "feature_ok"]].copy()
        assert_frame_equal(output_df, good_df)
        self.assertTrue(output_excluded_df.empty)

    def test_filter_on_column_with_inf(self):
        # Test that the function exclude columns where feature value is 'inf'
        data = pd.DataFrame({"feature_1": [1.5601, 0, 2.33, 11.32], "feature_ok": np.arange(1, 5)})

        data["feature_with_inf"] = 1 / data["feature_1"]
        data["id"] = np.arange(1, 5, dtype="int64")
        bad_df = data[np.isinf(data["feature_with_inf"])].copy()
        good_df = data[~np.isinf(data["feature_with_inf"])].copy()
        bad_df.reset_index(drop=True, inplace=True)
        good_df.reset_index(drop=True, inplace=True)

        output_df, output_excluded_df = self.fpp.filter_on_column(
            data, "feature_with_inf", "id", exclude_zeros=False, exclude_zero_sd=True
        )

        assert_frame_equal(output_df, good_df)
        assert_frame_equal(output_excluded_df, bad_df)

    def test_filter_on_flag_column_empty_flag_dictionary(self):
        # no flags specified, keep the data frame as is
        df = pd.DataFrame(
            {
                "spkitemid": ["a", "b", "c", "d"],
                "sc1": [1, 2, 1, 3],
                "feature": [2, 3, 4, 5],
                "flag1": [0, 0, 0, 0],
                "flag2": [1, 2, 2, 1],
            }
        )
        flag_dict = {}

        df_new, df_excluded = self.fpp.filter_on_flag_columns(df, flag_dict)
        assert_frame_equal(df_new, df)
        self.assertEqual(len(df_excluded), 0)

    def test_filter_on_flag_column_nothing_to_exclude_int_column_and_dict(self):
        df = pd.DataFrame(
            {
                "spkitemid": ["a", "b", "c", "d"],
                "sc1": [1, 2, 1, 3],
                "feature": [2, 3, 4, 5],
                "flag1": [0, 1, 2, 3],
            }
        )
        flag_dict = {"flag1": [0, 1, 2, 3, 4]}

        df_new, df_excluded = self.fpp.filter_on_flag_columns(df, flag_dict)
        assert_frame_equal(df_new, df)
        self.assertEqual(len(df_excluded), 0)

    def test_filter_on_flag_column_nothing_to_exclude_float_column_and_dict(self):
        df = pd.DataFrame(
            {
                "spkitemid": ["a", "b", "c", "d"],
                "sc1": [1, 2, 1, 3],
                "feature": [2, 3, 4, 5],
                "flag1": [0.5, 1.1, 2.2, 3.6],
            }
        )
        flag_dict = {"flag1": [0.5, 1.1, 2.2, 3.6, 4.5]}

        df_new, df_excluded = self.fpp.filter_on_flag_columns(df, flag_dict)
        assert_frame_equal(df_new, df)
        self.assertEqual(len(df_excluded), 0)

    def test_filter_on_flag_column_nothing_to_exclude_str_column_and_dict(self):
        df = pd.DataFrame(
            {
                "spkitemid": ["a", "b", "c", "d"],
                "sc1": [1, 2, 1, 3],
                "feature": [2, 3, 4, 5],
                "flag1": ["a", "b", "c", "d"],
            }
        )
        flag_dict = {"flag1": ["a", "b", "c", "d", "e"]}

        df_new, df_excluded = self.fpp.filter_on_flag_columns(df, flag_dict)
        assert_frame_equal(df_new, df)
        self.assertEqual(len(df_excluded), 0)

    def test_filter_on_flag_column_nothing_to_exclude_float_column_int_dict(self):
        df = pd.DataFrame(
            {
                "spkitemid": ["a", "b", "c", "d"],
                "sc1": [1, 2, 1, 3],
                "feature": [2, 3, 4, 5],
                "flag1": [0.0, 1.0, 2.0, 3.0],
            }
        )
        flag_dict = {"flag1": [0, 1, 2, 3, 4]}

        df_new, df_excluded = self.fpp.filter_on_flag_columns(df, flag_dict)
        assert_frame_equal(df_new, df)
        self.assertEqual(len(df_excluded), 0)

    def test_filter_on_flag_column_nothing_to_exclude_int_column_float_dict(self):
        df = pd.DataFrame(
            {
                "spkitemid": ["a", "b", "c", "d"],
                "sc1": [1, 2, 1, 3],
                "feature": [2, 3, 4, 5],
                "flag1": [0, 1, 2, 3],
            }
        )
        flag_dict = {"flag1": [0.0, 1.0, 2.0, 3.0, 4.5]}

        df_new, df_excluded = self.fpp.filter_on_flag_columns(df, flag_dict)
        assert_frame_equal(df_new, df)
        self.assertEqual(len(df_excluded), 0)

    def test_filter_on_flag_column_nothing_to_exclude_str_column_float_dict(self):
        df = pd.DataFrame(
            {
                "spkitemid": ["a", "b", "c", "d"],
                "sc1": [1, 2, 1, 3],
                "feature": [2, 3, 4, 5],
                "flag1": ["4", "1", "2", "3.5"],
            }
        )
        flag_dict = {"flag1": [0.0, 1.0, 2.0, 3.5, 4.0]}

        df_new, df_excluded = self.fpp.filter_on_flag_columns(df, flag_dict)
        assert_frame_equal(df_new, df)
        self.assertEqual(len(df_excluded), 0)

    def test_filter_on_flag_column_nothing_to_exclude_float_column_str_dict(self):
        df = pd.DataFrame(
            {
                "spkitemid": ["a", "b", "c", "d"],
                "sc1": [1, 2, 1, 3],
                "feature": [2, 3, 4, 5],
                "flag1": [4.0, 1.0, 2.0, 3.5],
            }
        )
        flag_dict = {"flag1": ["1", "2", "3.5", "4", "TD"]}

        df_new, df_excluded = self.fpp.filter_on_flag_columns(df, flag_dict)
        assert_frame_equal(df_new, df)
        self.assertEqual(len(df_excluded), 0)

    def test_filter_on_flag_column_nothing_to_exclude_str_column_int_dict(self):
        df = pd.DataFrame(
            {
                "spkitemid": ["a", "b", "c", "d"],
                "sc1": [1, 2, 1, 3],
                "feature": [2, 3, 4, 5],
                "flag1": ["0.0", "1.0", "2.0", "3.0"],
            }
        )
        flag_dict = {"flag1": [0, 1, 2, 3, 4]}

        df_new, df_excluded = self.fpp.filter_on_flag_columns(df, flag_dict)
        assert_frame_equal(df_new, df)
        self.assertEqual(len(df_excluded), 0)

    def test_filter_on_flag_column_nothing_to_exclude_int_column_str_dict(self):
        df = pd.DataFrame(
            {
                "spkitemid": ["a", "b", "c", "d"],
                "sc1": [1, 2, 1, 3],
                "feature": [2, 3, 4, 5],
                "flag1": [0, 1, 2, 3],
            }
        )
        flag_dict = {"flag1": ["0.0", "1.0", "2.0", "3.0", "TD"]}

        df_new, df_excluded = self.fpp.filter_on_flag_columns(df, flag_dict)
        assert_frame_equal(df_new, df)
        self.assertEqual(len(df_excluded), 0)

    def test_filter_on_flag_column_nothing_to_exclude_mixed_type_column_str_dict(self):
        df = pd.DataFrame(
            {
                "spkitemid": ["a", "b", "c", "d"],
                "sc1": [1, 2, 1, 3],
                "feature": [2, 3, 4, 5],
                "flag1": [0, "1.0", 2, 3.5],
            }
        )
        flag_dict = {"flag1": ["0.0", "1.0", "2.0", "3.5", "TD"]}

        df_new, df_excluded = self.fpp.filter_on_flag_columns(df, flag_dict)
        assert_frame_equal(df_new, df)
        self.assertEqual(len(df_excluded), 0)

    def test_filter_on_flag_column_nothing_to_exclude_mixed_type_column_int_dict(self):
        df = pd.DataFrame(
            {
                "spkitemid": ["a", "b", "c", "d"],
                "sc1": [1, 2, 1, 3],
                "feature": [2, 3, 4, 5],
                "flag1": [0, "1.0", 2, 3.0],
            }
        )
        flag_dict = {"flag1": [0, 1, 2, 3, 4]}

        df_new, df_excluded = self.fpp.filter_on_flag_columns(df, flag_dict)
        assert_frame_equal(df_new, df)
        self.assertEqual(len(df_excluded), 0)

    def test_filter_on_flag_column_nothing_to_exclude_mixed_type_column_float_dict(
        self,
    ):
        df = pd.DataFrame(
            {
                "spkitemid": ["a", "b", "c", "d"],
                "sc1": [1, 2, 1, 3],
                "feature": [2, 3, 4, 5],
                "flag1": [0, "1.5", 2, 3.5],
            }
        )
        flag_dict = {"flag1": [0.0, 1.5, 2.0, 3.5, 4.0]}

        df_new, df_excluded = self.fpp.filter_on_flag_columns(df, flag_dict)
        assert_frame_equal(df_new, df)
        self.assertEqual(len(df_excluded), 0)

    def test_filter_on_flag_column_nothing_to_exclude_int_column_mixed_type_dict(self):
        df = pd.DataFrame(
            {
                "spkitemid": ["a", "b", "c", "d"],
                "sc1": [1, 2, 1, 3],
                "feature": [2, 3, 4, 5],
                "flag1": [0, 1, 2, 3],
            }
        )
        flag_dict = {"flag1": [0, 1, 2, 3.0, 3.5, "TD"]}

        df_new, df_excluded = self.fpp.filter_on_flag_columns(df, flag_dict)
        assert_frame_equal(df_new, df)
        self.assertEqual(len(df_excluded), 0)

    def test_filter_on_flag_column_nothing_to_exclude_float_column_mixed_type_dict(
        self,
    ):
        df = pd.DataFrame(
            {
                "spkitemid": ["a", "b", "c", "d"],
                "sc1": [1, 2, 1, 3],
                "feature": [2, 3, 4, 5],
                "flag1": [0.0, 1.0, 2.0, 3.5],
            }
        )
        flag_dict = {"flag1": [0, 1, 2, 3.0, 3.5, "TD"]}

        df_new, df_excluded = self.fpp.filter_on_flag_columns(df, flag_dict)
        assert_frame_equal(df_new, df)
        self.assertEqual(len(df_excluded), 0)

    def test_filter_on_flag_column_nothing_to_exclude_str_column_mixed_type_dict(self):
        df = pd.DataFrame(
            {
                "spkitemid": ["a", "b", "c", "d"],
                "sc1": [1, 2, 1, 3],
                "feature": [2, 3, 4, 5],
                "flag1": ["0.0", "1.0", "2.0", "3.5"],
            }
        )
        flag_dict = {"flag1": [0, 1, 2, 3.0, 3.5, "TD"]}

        df_new, df_excluded = self.fpp.filter_on_flag_columns(df, flag_dict)
        assert_frame_equal(df_new, df)
        self.assertEqual(len(df_excluded), 0)

    def test_filter_on_flag_column_nothing_to_exclude_mixed_type_column_mixed_type_dict(
        self,
    ):
        df = pd.DataFrame(
            {
                "spkitemid": ["a", "b", "c", "d"],
                "sc1": [1, 2, 1, 3],
                "feature": [2, 3, 4, 5],
                "flag1": [1, 2, 3.5, "TD"],
            }
        )
        flag_dict = {"flag1": [0, 1, 2, 3.0, 3.5, "TD"]}

        df_new, df_excluded = self.fpp.filter_on_flag_columns(df, flag_dict)
        assert_frame_equal(df_new, df)
        self.assertEqual(len(df_excluded), 0)

    def test_filter_on_flag_column_mixed_type_column_mixed_type_dict_filter_preserve_type(
        self,
    ):
        df = pd.DataFrame(
            {
                "spkitemid": ["a", "b", "c", "d", "e", "f"],
                "sc1": [1, 2, 1, 3, 4, 5],
                "feature": [2, 3, 4, 5, 6, 2],
                "flag1": [1, 1.5, 2, 3.5, "TD", "NS"],
            }
        )
        flag_dict = {"flag1": [1.5, 2, "TD"]}

        df_new_expected = pd.DataFrame(
            {
                "spkitemid": ["b", "c", "e"],
                "sc1": [2, 1, 4],
                "feature": [3, 4, 6],
                "flag1": [1.5, 2, "TD"],
            }
        )

        df_excluded_expected = pd.DataFrame(
            {
                "spkitemid": ["a", "d", "f"],
                "sc1": [1, 3, 5],
                "feature": [2, 5, 2],
                "flag1": [1, 3.5, "NS"],
            }
        )

        df_new, df_excluded = self.fpp.filter_on_flag_columns(df, flag_dict)
        assert_frame_equal(df_new, df_new_expected)
        assert_frame_equal(df_excluded, df_excluded_expected)

    def test_filter_on_flag_column_with_none_value_in_int_flag_column_int_dict(self):
        df = pd.DataFrame(
            {
                "spkitemid": [1, 2, 3, 4, 5, 6],
                "sc1": [1, 2, 1, 3, 4, 5],
                "feature": [2, 3, 4, 5, 6, 2],
                "flag1": [1, 2, 2, 3, 4, None],
            },
            dtype=object,
        )

        flag_dict = {"flag1": [2, 4]}

        df_new_expected = pd.DataFrame(
            {
                "spkitemid": [2, 3, 5],
                "sc1": [2, 1, 4],
                "feature": [3, 4, 6],
                "flag1": [2, 2, 4],
            },
            dtype=object,
        )

        df_excluded_expected = pd.DataFrame(
            {
                "spkitemid": [1, 4, 6],
                "sc1": [1, 3, 5],
                "feature": [2, 5, 2],
                "flag1": [1, 3, None],
            },
            dtype=object,
        )

        df_new, df_excluded = self.fpp.filter_on_flag_columns(df, flag_dict)
        assert_frame_equal(df_new, df_new_expected)
        assert_frame_equal(df_excluded, df_excluded_expected)

    def test_filter_on_flag_column_with_none_value_in_float_flag_column_float_dict(
        self,
    ):
        df = pd.DataFrame(
            {
                "spkitemid": ["a", "b", "c", "d", "e", "f"],
                "sc1": [1, 2, 1, 3, 4, 5],
                "feature": [2, 3, 4, 5, 6, 2],
                "flag1": [1.2, 2.1, 2.1, 3.3, 4.2, None],
            }
        )
        flag_dict = {"flag1": [2.1, 4.2]}

        df_new_expected = pd.DataFrame(
            {
                "spkitemid": ["b", "c", "e"],
                "sc1": [2, 1, 4],
                "feature": [3, 4, 6],
                "flag1": [2.1, 2.1, 4.2],
            }
        )

        df_excluded_expected = pd.DataFrame(
            {
                "spkitemid": ["a", "d", "f"],
                "sc1": [1, 3, 5],
                "feature": [2, 5, 2],
                "flag1": [1.2, 3.3, None],
            }
        )

        df_new, df_excluded = self.fpp.filter_on_flag_columns(df, flag_dict)
        assert_frame_equal(df_new, df_new_expected)
        assert_frame_equal(df_excluded, df_excluded_expected)

    def test_filter_on_flag_column_with_none_value_in_str_flag_column_str_dict(self):
        df = pd.DataFrame(
            {
                "spkitemid": ["a", "b", "c", "d", "e", "f"],
                "sc1": [1, 2, 1, 3, 4, 5],
                "feature": [2, 3, 4, 5, 6, 2],
                "flag1": ["a", "b", "b", "c", "d", None],
            }
        )
        flag_dict = {"flag1": ["b", "d"]}

        df_new_expected = pd.DataFrame(
            {
                "spkitemid": ["b", "c", "e"],
                "sc1": [2, 1, 4],
                "feature": [3, 4, 6],
                "flag1": ["b", "b", "d"],
            }
        )

        df_excluded_expected = pd.DataFrame(
            {
                "spkitemid": ["a", "d", "f"],
                "sc1": [1, 3, 5],
                "feature": [2, 5, 2],
                "flag1": ["a", "c", None],
            }
        )

        df_new, df_excluded = self.fpp.filter_on_flag_columns(df, flag_dict)
        assert_frame_equal(df_new, df_new_expected)
        assert_frame_equal(df_excluded, df_excluded_expected)

    def test_filter_on_flag_column_with_none_value_in_mixed_type_flag_column_float_dict(
        self,
    ):
        df = pd.DataFrame(
            {
                "spkitemid": ["a", "b", "c", "d", "e", "f"],
                "sc1": [1, 2, 1, 3, 4, 5],
                "feature": [2, 3, 4, 5, 6, 2],
                "flag1": [1, 1.5, 2.0, "TD", 2.0, None],
            },
            dtype=object,
        )
        flag_dict = {"flag1": [1.5, 2.0]}

        df_new_expected = pd.DataFrame(
            {
                "spkitemid": ["b", "c", "e"],
                "sc1": [2, 1, 4],
                "feature": [3, 4, 6],
                "flag1": [1.5, 2.0, 2.0],
            },
            dtype=object,
        )

        df_excluded_expected = pd.DataFrame(
            {
                "spkitemid": ["a", "d", "f"],
                "sc1": [1, 3, 5],
                "feature": [2, 5, 2],
                "flag1": [1, "TD", None],
            },
            dtype=object,
        )

        df_new, df_excluded = self.fpp.filter_on_flag_columns(df, flag_dict)
        assert_frame_equal(df_new, df_new_expected)
        assert_frame_equal(df_excluded, df_excluded_expected)

    def test_filter_on_flag_column_with_none_value_in_mixed_type_flag_column_int_dict(
        self,
    ):
        df = pd.DataFrame(
            {
                "spkitemid": ["a", "b", "c", "d", "e", "f"],
                "sc1": [1, 2, 1, 3, 4, 5],
                "feature": [2, 3, 4, 5, 6, 2],
                "flag1": [1.5, 2, 2, "TD", 4, None],
            },
            dtype=object,
        )
        flag_dict = {"flag1": [2, 4]}

        df_new_expected = pd.DataFrame(
            {
                "spkitemid": ["b", "c", "e"],
                "sc1": [2, 1, 4],
                "feature": [3, 4, 6],
                "flag1": [2, 2, 4],
            },
            dtype=object,
        )

        df_excluded_expected = pd.DataFrame(
            {
                "spkitemid": ["a", "d", "f"],
                "sc1": [1, 3, 5],
                "feature": [2, 5, 2],
                "flag1": [1.5, "TD", None],
            },
            dtype=object,
        )

        df_new, df_excluded = self.fpp.filter_on_flag_columns(df, flag_dict)
        assert_frame_equal(df_new, df_new_expected)
        assert_frame_equal(df_excluded, df_excluded_expected)

    def test_filter_on_flag_column_with_none_value_in_mixed_type_flag_column_mixed_type_dict(
        self,
    ):
        df = pd.DataFrame(
            {
                "spkitemid": ["a", "b", "c", "d", "e", "f"],
                "sc1": [1, 2, 1, 3, 4, 5],
                "feature": [2, 3, 4, 5, 6, 2],
                "flag1": [1, 1.5, 2, 3.5, "TD", None],
            },
            dtype=object,
        )
        flag_dict = {"flag1": [1.5, 2, "TD"]}

        df_new_expected = pd.DataFrame(
            {
                "spkitemid": ["b", "c", "e"],
                "sc1": [2, 1, 4],
                "feature": [3, 4, 6],
                "flag1": [1.5, 2, "TD"],
            },
            dtype=object,
        )

        df_excluded_expected = pd.DataFrame(
            {
                "spkitemid": ["a", "d", "f"],
                "sc1": [1, 3, 5],
                "feature": [2, 5, 2],
                "flag1": [1, 3.5, None],
            },
            dtype=object,
        )

        df_new, df_excluded = self.fpp.filter_on_flag_columns(df, flag_dict)
        assert_frame_equal(df_new, df_new_expected)
        assert_frame_equal(df_excluded, df_excluded_expected)

    def test_filter_on_flag_column_two_flags_same_responses(self):
        df = pd.DataFrame(
            {
                "spkitemid": ["a", "b", "c", "d", "e", "f"],
                "sc1": [1, 2, 1, 3, 4, 5],
                "feature": [2, 3, 4, 5, 6, 2],
                "flag1": [1, 1.5, 2, 3.5, "TD", "NS"],
                "flag2": [1, 0, 0, 1, 0, 1],
            }
        )
        flag_dict = {"flag1": [1.5, 2, "TD"], "flag2": [0]}

        df_new_expected = pd.DataFrame(
            {
                "spkitemid": ["b", "c", "e"],
                "sc1": [2, 1, 4],
                "feature": [3, 4, 6],
                "flag1": [1.5, 2, "TD"],
                "flag2": [0, 0, 0],
            }
        )

        df_excluded_expected = pd.DataFrame(
            {
                "spkitemid": ["a", "d", "f"],
                "sc1": [1, 3, 5],
                "feature": [2, 5, 2],
                "flag1": [1, 3.5, "NS"],
                "flag2": [1, 1, 1],
            }
        )

        df_new, df_excluded = self.fpp.filter_on_flag_columns(df, flag_dict)
        assert_frame_equal(df_new, df_new_expected)
        assert_frame_equal(df_excluded, df_excluded_expected)

    def test_filter_on_flag_column_two_flags_different_responses(self):
        df = pd.DataFrame(
            {
                "spkitemid": ["a", "b", "c", "d", "e", "f"],
                "sc1": [1, 2, 1, 3, 4, 5],
                "feature": [2, 3, 4, 5, 6, 2],
                "flag1": [1, 1.5, 2, 3.5, "TD", "NS"],
                "flag2": [2, 0, 0, 1, 0, 1],
            }
        )
        flag_dict = {"flag1": [1.5, 2, "TD", "NS"], "flag2": [0, 2]}

        df_new_expected = pd.DataFrame(
            {
                "spkitemid": ["b", "c", "e"],
                "sc1": [2, 1, 4],
                "feature": [3, 4, 6],
                "flag1": [1.5, 2, "TD"],
                "flag2": [0, 0, 0],
            }
        )

        df_excluded_expected = pd.DataFrame(
            {
                "spkitemid": ["a", "d", "f"],
                "sc1": [1, 3, 5],
                "feature": [2, 5, 2],
                "flag1": [1, 3.5, "NS"],
                "flag2": [2, 1, 1],
            }
        )

        df_new, df_excluded = self.fpp.filter_on_flag_columns(df, flag_dict)
        assert_frame_equal(df_new, df_new_expected)
        assert_frame_equal(df_excluded, df_excluded_expected)

    def test_filter_on_flag_column_missing_columns(self):
        df = pd.DataFrame(
            {
                "spkitemid": ["a", "b", "c", "d"],
                "sc1": [1, 2, 1, 3],
                "feature": [2, 3, 4, 5],
                "flag1": ["1", "1", "1", "1"],
                "flag2": ["1", "2", "2", "1"],
            }
        )
        flag_dict = {"flag3": ["0"], "flag2": ["1", "2"]}

        with self.assertRaises(KeyError):
            df_new, df_excluded = self.fpp.filter_on_flag_columns(df, flag_dict)

    def test_filter_on_flag_column_nothing_left(self):
        bad_df = pd.DataFrame(
            {
                "spkitemid": ["a1", "b1", "c1", "d1"],
                "sc1": [1, 2, 1, 3],
                "feature": [2, 3, 4, 5],
                "flag1": [1, 0, 20, 14],
                "flag2": [1, 1.0, "TD", "03"],
            }
        )

        flag_dict = {"flag1": [1, 0, 14], "flag2": ["TD"]}

        with self.assertRaises(ValueError):
            df_new, df_excluded = self.fpp.filter_on_flag_columns(bad_df, flag_dict)

    def test_remove_outliers(self):
        # we want to test that even if we pass in a list of
        # integers, we still get the right clamped output
        data = [1, 1, 2, 2, 1, 1] * 10 + [10]
        ceiling = np.mean(data) + 4 * np.std(data)

        clamped_data = self.fpp.remove_outliers(data)
        self.assertAlmostEqual(clamped_data[-1], ceiling)

    def test_generate_feature_names_subset(self):
        reserved_column_names = ["reserved_col1", "reserved_col2"]
        expected = ["col_1"]

        df = pd.DataFrame(
            {
                "reserved_col1": ["X", "Y", "Z"],
                "reserved_col2": ["Q", "R", "S"],
                "col_1": [1, 2, 3],
                "col_2": ["A", "B", "C"],
            }
        )
        subset = "A"

        feature_subset = pd.DataFrame(
            {"Feature": ["col_1", "col_2", "col_3"], "A": [1, 0, 0], "B": [1, 1, 1]}
        )

        feat_names = self.fpp.generate_feature_names(
            df, reserved_column_names, feature_subset, subset
        )
        self.assertEqual(feat_names, expected)

    def test_generate_feature_names_none(self):
        reserved_column_names = ["reserved_col1", "reserved_col2"]
        expected = ["col_1", "col_2"]

        df = pd.DataFrame(
            {
                "reserved_col1": ["X", "Y", "Z"],
                "reserved_col2": ["Q", "R", "S"],
                "col_1": [1, 2, 3],
                "col_2": ["A", "B", "C"],
            }
        )

        feat_names = self.fpp.generate_feature_names(
            df, reserved_column_names, feature_subset_specs=None, feature_subset=None
        )
        self.assertEqual(feat_names, expected)

    def test_model_name_builtin_model(self):
        model_name = "LinearRegression"
        model_type = self.fpp.check_model_name(model_name)
        self.assertEqual(model_type, "BUILTIN")

    def test_model_name_skll_model(self):
        model_name = "AdaBoostRegressor"
        model_type = self.fpp.check_model_name(model_name)
        self.assertEqual(model_type, "SKLL")

    def test_model_name_wrong_name(self):
        model_name = "random_model"
        with self.assertRaises(ValueError):
            self.fpp.check_model_name(model_name)

    def test_trim(self):
        values = np.array([1.4, 8.5, 7.4])
        expected = np.array([1.4, 8.4998, 7.4])
        actual = self.fpp.trim(values, 1, 8)
        assert_array_equal(actual, expected)

    def test_trim_with_list(self):
        values = [1.4, 8.5, 7.4]
        expected = np.array([1.4, 8.4998, 7.4])
        actual = self.fpp.trim(values, 1, 8)
        assert_array_equal(actual, expected)

    def test_trim_with_custom_tolerance(self):
        values = [0.6, 8.4, 7.4]
        expected = np.array([0.75, 8.25, 7.4])
        actual = self.fpp.trim(values, 1, 8, 0.25)
        assert_array_equal(actual, expected)

    def test_preprocess_feature_fail(self):
        np.random.seed(10)
        values = np.random.random(size=1000)
        values = np.append(values, np.array([10000000]))

        mean = values.mean()
        std = values.std()

        expected = values.copy()
        expected[-1] = mean + 4 * std

        actual = self.fpp.preprocess_feature(values, "A", "raw", mean, std)

        assert_array_equal(actual, expected)

    def test_preprocess_feature_with_outlier(self):
        np.random.seed(10)
        values = np.random.random(size=1000)
        values = np.append(values, np.array([10000000]))

        mean = values.mean()
        std = values.std()

        expected = values.copy()
        expected[-1] = mean + 4 * std

        actual = self.fpp.preprocess_feature(values, "A", "raw", mean, std, exclude_zero_sd=True)

        assert_array_equal(actual, expected)

    def test_preprocess_features(self):
        train = pd.DataFrame({"A": [1, 2, 4, 3]})
        test = pd.DataFrame({"A": [4, 3, 2, 1]})

        train_expected = (train["A"] - train["A"].mean()) / train["A"].std()
        train_expected = pd.DataFrame(train_expected)

        test_expected = (test["A"] - test["A"].mean()) / test["A"].std()
        test_expected = pd.DataFrame(test_expected)

        info_expected = pd.DataFrame(
            {
                "feature": ["A"],
                "sign": [1],
                "train_mean": [train.A.mean()],
                "train_sd": [train.A.std()],
                "train_transformed_mean": [train.A.mean()],
                "train_transformed_sd": [test.A.std()],
                "transform": ["raw"],
            }
        )

        specs = pd.DataFrame({"feature": ["A"], "transform": ["raw"], "sign": [1]})

        (
            train_processed,
            test_processed,
            info_processed,
        ) = self.fpp.preprocess_features(train, test, specs)

        assert_frame_equal(train_processed.sort_index(axis=1), train_expected.sort_index(axis=1))
        assert_frame_equal(test_processed.sort_index(axis=1), test_expected.sort_index(axis=1))
        assert_frame_equal(info_processed.sort_index(axis=1), info_expected.sort_index(axis=1))

    def test_filter_data_features(self):
        data = {
            "ID": [1, 2, 3, 4],
            "LENGTH": [10, 12, 11, 12],
            "h1": [1, 2, 3, 1],
            "candidate": ["A", "B", "C", "A"],
            "h2": [1, 2, 3, 1],
            "feature1": [1, 3, 4, 1],
            "feature2": [1, 3, 2, 2],
        }

        df_filtered_features_expected = pd.DataFrame(
            {
                "spkitemid": [1, 2, 3, 4],
                "sc1": [1.0, 2.0, 3.0, 1.0],
                "feature1": [1.0, 3.0, 4.0, 1.0],
                "feature2": [1.0, 3.0, 2.0, 2.0],
            }
        )
        df_filtered_features_expected = df_filtered_features_expected[
            ["spkitemid", "sc1", "feature1", "feature2"]
        ]

        data = pd.DataFrame(data)

        (df_filtered_features, _, _, _, _, _, _, _, _, _) = self.fpp.filter_data(
            data,
            "h1",
            "ID",
            "LENGTH",
            "h2",
            "candidate",
            ["feature1", "feature2"],
            ["LENGTH", "ID", "candidate", "h1"],
            0,
            6,
            {},
            [],
        )

        assert_frame_equal(df_filtered_features, df_filtered_features_expected)

    def test_filter_data_correct_features_and_length_in_other_columns(self):
        data = {
            "ID": [1, 2, 3, 4],
            "LENGTH": [10, 10, 10, 10],
            "h1": [1, 2, 3, 1],
            "candidate": ["A", "B", "C", "A"],
            "h2": [1, 2, 3, 1],
            "feature1": [1, 3, 4, 1],
            "feature2": [1, 3, 2, 2],
        }

        data = pd.DataFrame(data)

        (
            _,
            _,
            df_filtered_other_columns,
            _,
            _,
            _,
            _,
            _,
            _,
            feature_names,
        ) = self.fpp.filter_data(
            data,
            "h1",
            "ID",
            "LENGTH",
            "h2",
            "candidate",
            ["feature1", "feature2"],
            ["LENGTH", "ID", "candidate", "h1"],
            0,
            6,
            {},
            [],
        )

        self.assertEqual(feature_names, ["feature1", "feature2"])
        assert "##LENGTH##" in df_filtered_other_columns.columns

    def test_filter_data_length_in_other_columns(self):
        data = {
            "ID": [1, 2, 3, 4],
            "LENGTH": [10, 10, 10, 10],
            "h1": [1, 2, 3, 1],
            "candidate": ["A", "B", "C", "A"],
            "h2": [1, 2, 3, 1],
            "feature1": [1, 3, 4, 1],
            "feature2": [1, 3, 2, 2],
        }

        data = pd.DataFrame(data)

        (
            _,
            _,
            df_filtered_other_columns,
            _,
            _,
            _,
            _,
            _,
            _,
            feature_names,
        ) = self.fpp.filter_data(
            data,
            "h1",
            "ID",
            "LENGTH",
            "h2",
            "candidate",
            ["feature1", "feature2"],
            ["LENGTH", "ID", "candidate", "h1"],
            0,
            6,
            {},
            [],
        )

        self.assertEqual(feature_names, ["feature1", "feature2"])
        assert "##LENGTH##" in df_filtered_other_columns.columns

    def test_filter_data_min_candidates_raises_value_error(self):
        data = {
            "ID": [1, 2, 3, 4],
            "LENGTH": [10, 10, 10, 10],
            "h1": [1, 2, 3, 1],
            "candidate": ["A", "B", "C", "A"],
            "h2": [1, 2, 3, 1],
            "feature1": [1, 3, 4, 1],
            "feature2": [1, 3, 2, 2],
        }

        data = pd.DataFrame(data)

        with self.assertRaises(ValueError):
            self.fpp.filter_data(
                data,
                "h1",
                "ID",
                "LENGTH",
                "h2",
                "candidate",
                ["feature1", "feature2"],
                ["LENGTH", "ID", "candidate", "h1"],
                0,
                6,
                {},
                [],
                min_candidate_items=5,
            )

    def test_filter_data_with_min_candidates(self):
        data = {
            "ID": [1, 2, 3, 4],
            "LENGTH": [10, 10, 10, 10],
            "h1": [1, 2, 3, 1],
            "candidate": ["A", "B", "C", "A"],
            "h2": [1, 2, 3, 1],
            "feature1": [1, 3, 4, 1],
            "feature2": [1, 3, 2, 2],
        }

        data = pd.DataFrame(data)

        (
            df_filtered_features,
            _,
            _,
            _,
            _,
            df_filtered_human_scores,
            _,
            _,
            _,
            _,
        ) = self.fpp.filter_data(
            data,
            "h1",
            "ID",
            "LENGTH",
            "h2",
            "candidate",
            ["feature1", "feature2"],
            ["LENGTH", "ID", "candidate", "h1"],
            0,
            6,
            {},
            [],
            min_candidate_items=2,
        )

        self.assertEqual(df_filtered_features.shape[0], 2)
        assert all(col in df_filtered_human_scores.columns for col in ["sc1", "sc2"])

    def test_filter_data_id_candidate_equal(self):
        data = {
            "LENGTH": [10, 12, 18, 21],
            "h1": [1, 2, 3, 1],
            "candidate": ["A", "B", "C", "D"],
            "h2": [1, 2, 3, 1],
            "feature1": [1, 3, 4, 1],
            "feature2": [1, 3, 2, 2],
        }

        data = pd.DataFrame(data)

        (_, df_filtered_metadata, _, _, _, _, _, _, _, _) = self.fpp.filter_data(
            data,
            "h1",
            "candidate",
            "LENGTH",
            "h2",
            "candidate",
            ["feature1", "feature2"],
            ["LENGTH", "ID", "candidate", "h1"],
            0,
            6,
            {},
            [],
        )

        expected = pd.DataFrame(
            {"spkitemid": ["A", "B", "C", "D"], "candidate": ["A", "B", "C", "D"]}
        )
        expected = expected[["spkitemid", "candidate"]]
        assert_frame_equal(df_filtered_metadata, expected)


class TestFeatureSpecsProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fsp = FeatureSpecsProcessor()

    def test_generate_default_specs(self):
        fnames = ["Grammar", "Vocabulary", "Pronunciation"]
        df_specs = self.fsp.generate_default_specs(fnames)
        self.assertEqual(len(df_specs), 3)
        self.assertEqual(df_specs["feature"][0], "Grammar")
        self.assertEqual(df_specs["transform"][1], "raw")
        self.assertEqual(df_specs["sign"][2], 1.0)

    def test_generate_specs_from_data_with_negative_sign(self):
        feature_subset_specs = pd.DataFrame(
            {
                "Feature": [
                    "Grammar",
                    "Vocabulary",
                    "Fluency",
                    "Content_coverage",
                    "Discourse",
                ],
                "Sign_SYS1": ["-", "+", "+", "+", "-"],
            }
        )

        np.random.seed(10)
        data = {
            "Grammar": np.random.randn(10),
            "Fluency": np.random.randn(10),
            "Discourse": np.random.randn(10),
            "r1": np.random.choice(4, 10),
            "spkitemlab": ["a-5"] * 10,
        }
        df = pd.DataFrame(data)

        df_specs = self.fsp.generate_specs(
            df, ["Grammar", "Fluency", "Discourse"], "r1", feature_subset_specs, "SYS1"
        )

        self.assertEqual(len(df_specs), 3)
        assert_array_equal(df_specs["feature"], ["Grammar", "Fluency", "Discourse"])
        assert_array_equal(df_specs["sign"], [-1.0, 1.0, -1.0])

    def test_generate_specs_from_data_with_default_sign(self):
        feature_subset_specs = pd.DataFrame(
            {
                "Feature": [
                    "Grammar",
                    "Vocabulary",
                    "Fluency",
                    "Content_coverage",
                    "Discourse",
                ],
                "Sign_SYS1": ["-", "+", "+", "+", "-"],
            }
        )

        np.random.seed(10)
        data = {
            "Grammar": np.random.randn(10),
            "Fluency": np.random.randn(10),
            "Discourse": np.random.randn(10),
            "r1": np.random.choice(4, 10),
            "spkitemlab": ["a-5"] * 10,
        }
        df = pd.DataFrame(data)
        df_specs = self.fsp.generate_specs(
            df,
            ["Grammar", "Fluency", "Discourse"],
            "r1",
            feature_subset_specs,
            feature_sign=None,
        )
        self.assertEqual(len(df_specs), 3)
        assert_array_equal(df_specs["feature"], ["Grammar", "Fluency", "Discourse"])
        assert_array_equal(df_specs["sign"], [1.0, 1.0, 1.0])

    def test_generate_specs_from_data_with_transformation(self):
        feature_subset_specs = pd.DataFrame(
            {
                "Feature": [
                    "Grammar",
                    "Vocabulary",
                    "Fluency",
                    "Content_coverage",
                    "Discourse",
                ],
                "Sign_SYS1": ["-", "+", "+", "+", "-"],
            }
        )
        np.random.seed(10)
        r1 = np.random.choice(range(1, 5), 10)
        data = {
            "Grammar": np.random.randn(10),
            "Vocabulary": r1**2,
            "Discourse": np.random.randn(10),
            "r1": r1,
            "spkitemlab": ["a-5"] * 10,
        }
        df = pd.DataFrame(data)
        df_specs = self.fsp.generate_specs(
            df,
            ["Grammar", "Vocabulary", "Discourse"],
            "r1",
            feature_subset_specs,
            "SYS1",
        )
        assert_array_equal(df_specs["feature"], ["Grammar", "Vocabulary", "Discourse"])
        self.assertEqual(df_specs["transform"][1], "sqrt")

    def test_generate_specs_from_data_when_transformation_changes_sign(self):
        feature_subset_specs = pd.DataFrame(
            {
                "Feature": [
                    "Grammar",
                    "Vocabulary",
                    "Fluency",
                    "Content_coverage",
                    "Discourse",
                ],
                "Sign_SYS1": ["-", "+", "+", "+", "-"],
            }
        )
        np.random.seed(10)
        r1 = np.random.choice(range(1, 5), 10)
        data = {
            "Grammar": np.random.randn(10),
            "Vocabulary": 1 / r1,
            "Discourse": np.random.randn(10),
            "r1": r1,
            "spkitemlab": ["a-5"] * 10,
        }
        df = pd.DataFrame(data)
        df_specs = self.fsp.generate_specs(
            df,
            ["Grammar", "Vocabulary", "Discourse"],
            "r1",
            feature_subset_specs,
            "SYS1",
        )
        self.assertEqual(df_specs["feature"][1], "Vocabulary")
        self.assertEqual(df_specs["transform"][1], "addOneInv")
        self.assertEqual(df_specs["sign"][1], -1)

    def test_generate_specs_from_data_no_subset_specs(self):
        np.random.seed(10)
        data = {
            "Grammar": np.random.randn(10),
            "Fluency": np.random.randn(10),
            "Discourse": np.random.randn(10),
            "r1": np.random.choice(4, 10),
            "spkitemlab": ["a-5"] * 10,
        }
        df = pd.DataFrame(data)
        df_specs = self.fsp.generate_specs(df, ["Grammar", "Fluency", "Discourse"], "r1")
        self.assertEqual(len(df_specs), 3)
        assert_array_equal(df_specs["feature"], ["Grammar", "Fluency", "Discourse"])
        assert_array_equal(df_specs["sign"], [1.0, 1.0, 1.0])

    def test_validate_feature_specs(self):
        df_feature_specs = pd.DataFrame(
            {
                "feature": ["f1", "f2", "f3"],
                "sign": [1.0, 1.0, -1.0],
                "transform": ["raw", "inv", "sqrt"],
            }
        )

        df_new_feature_specs = self.fsp.validate_feature_specs(df_feature_specs)
        assert_frame_equal(df_feature_specs, df_new_feature_specs)

    def test_validate_feature_specs_with_Feature_as_column(self):
        df_feature_specs = pd.DataFrame(
            {
                "Feature": ["f1", "f2", "f3"],
                "sign": [1.0, 1.0, -1.0],
                "transform": ["raw", "inv", "sqrt"],
            }
        )
        df_expected_feature_specs = pd.DataFrame(
            {
                "feature": ["f1", "f2", "f3"],
                "sign": [1.0, 1.0, -1.0],
                "transform": ["raw", "inv", "sqrt"],
            }
        )

        df_new_feature_specs = self.fsp.validate_feature_specs(df_feature_specs)

        assert_frame_equal(df_new_feature_specs, df_expected_feature_specs)

    def test_validate_feature_specs_sign_to_float(self):
        df_feature_specs = pd.DataFrame(
            {
                "feature": ["f1", "f2", "f3"],
                "sign": ["1", "1", "-1"],
                "transform": ["raw", "inv", "sqrt"],
            }
        )
        df_expected_feature_specs = pd.DataFrame(
            {
                "feature": ["f1", "f2", "f3"],
                "sign": [1.0, 1.0, -1.0],
                "transform": ["raw", "inv", "sqrt"],
            }
        )

        df_new_feature_specs = self.fsp.validate_feature_specs(df_feature_specs)
        assert_frame_equal(df_new_feature_specs, df_expected_feature_specs)

    def test_validate_feature_specs_add_default_values(self):
        df_feature_specs = pd.DataFrame({"feature": ["f1", "f2", "f3"]})
        df_expected_feature_specs = pd.DataFrame(
            {
                "feature": ["f1", "f2", "f3"],
                "sign": [1, 1, 1],
                "transform": ["raw", "raw", "raw"],
            }
        )

        df_new_feature_specs = self.fsp.validate_feature_specs(df_feature_specs)
        assert_frame_equal(df_new_feature_specs, df_expected_feature_specs)

    def test_validate_feature_specs_wrong_sign_format(self):
        df_feature_specs = pd.DataFrame(
            {
                "feature": ["f1", "f2", "f3"],
                "sign": ["+", "+", "-"],
                "transform": ["raw", "inv", "sqrt"],
            }
        )

        with self.assertRaises(ValueError):
            self.fsp.validate_feature_specs(df_feature_specs)

    def test_validate_feature_duplicate_feature(self):
        df_feature_specs = pd.DataFrame(
            {
                "feature": ["f1", "f1", "f3"],
                "sign": ["+", "+", "-"],
                "transform": ["raw", "inv", "sqrt"],
            }
        )

        with self.assertRaises(ValueError):
            self.fsp.validate_feature_specs(df_feature_specs)

    def test_validate_feature_missing_feature_column(self):
        df_feature_specs = pd.DataFrame(
            {
                "FeatureName": ["f1", "f1", "f3"],
                "sign": ["+", "+", "-"],
                "transform": ["raw", "inv", "sqrt"],
            }
        )

        with self.assertRaises(KeyError):
            self.fsp.validate_feature_specs(df_feature_specs)


class TestFeatureSubsetProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fsp = FeatureSubsetProcessor()

    def test_select_by_subset(self):
        feature_subset_specs = pd.DataFrame(
            {
                "Feature": [
                    "Grammar",
                    "Vocabulary",
                    "Fluency",
                    "Content_coverage",
                    "Discourse",
                    "Pronunciation",
                    "Prosody",
                    "Content_accuracy",
                ],
                "high_entropy": [1, 1, 1, 1, 1, 1, 1, 0],
                "low_entropy": [0, 0, 1, 0, 0, 1, 1, 1],
            }
        )

        # This list should also trigger a warning about extra subset features not in the data
        fnames = ["Grammar", "Vocabulary", "Pronunciation", "Content_accuracy"]
        high_entropy_fnames = ["Grammar", "Vocabulary", "Pronunciation"]
        assert_array_equal(
            self.fsp.select_by_subset(fnames, feature_subset_specs, "high_entropy"),
            high_entropy_fnames,
        )

    def test_select_by_subset_warnings(self):
        feature_subset_specs = pd.DataFrame(
            {
                "Feature": [
                    "Grammar",
                    "Vocabulary",
                    "Fluency",
                    "Content_coverage",
                    "Discourse",
                    "Pronunciation",
                    "Prosody",
                    "Content_accuracy",
                ],
                "high_entropy": [1, 1, 1, 1, 1, 1, 1, 0],
                "low_entropy": [0, 0, 1, 0, 0, 1, 1, 1],
            }
        )

        extra_fnames = ["Grammar", "Vocabulary", "Rhythm"]
        assert_array_equal(
            self.fsp.select_by_subset(extra_fnames, feature_subset_specs, "high_entropy"),
            ["Grammar", "Vocabulary"],
        )

    def test_check_feature_subset_file_subset_only(self):
        feature_specs = pd.DataFrame({"feature": ["f1", "f2", "f3"], "subset1": [0, 1, 0]})
        self.fsp.check_feature_subset_file(feature_specs, "subset1")

    def test_check_feature_subset_file_sign_only(self):
        feature_specs = pd.DataFrame({"feature": ["f1", "f2", "f3"], "sign_SYS": ["+", "-", "+"]})
        self.fsp.check_feature_subset_file(feature_specs, sign="SYS")

    def test_check_feature_subset_file_sign_and_subset(self):
        feature_specs = pd.DataFrame(
            {
                "feature": ["f1", "f2", "f3"],
                "sign_SYS": ["+", "-", "+"],
                "subset1": [0, 1, 0],
            }
        )
        self.fsp.check_feature_subset_file(feature_specs, subset="subset1", sign="SYS")

    def test_check_feature_subset_file_sign_named_with_sign(self):
        feature_specs = pd.DataFrame({"feature": ["f1", "f2", "f3"], "sign_SYS": ["+", "-", "+"]})
        self.fsp.check_feature_subset_file(feature_specs, sign="SYS")

    def test_check_feature_subset_file_sign_named_with_Sign(self):
        feature_specs = pd.DataFrame({"feature": ["f1", "f2", "f3"], "Sign_SYS": ["+", "-", "+"]})
        self.fsp.check_feature_subset_file(feature_specs, sign="SYS")

    def test_check_feature_subset_file_sign_named_something_else(self):
        feature_specs = pd.DataFrame({"feature": ["f1", "f2", "f3"], "SYS_sign": ["+", "-", "+"]})
        with self.assertRaises(ValueError):
            self.fsp.check_feature_subset_file(feature_specs, sign="SYS")

    def test_check_feature_subset_file_multiple_sign_columns(self):
        feature_specs = pd.DataFrame(
            {
                "feature": ["f1", "f2", "f3"],
                "sign_SYS": ["+", "-", "+"],
                "Sign_SYS": ["-", "+", "-"],
            }
        )
        with self.assertRaises(ValueError):
            self.fsp.check_feature_subset_file(feature_specs, sign="SYS")

    def test_check_feature_subset_file_no_feature_column(self):
        feature_specs = pd.DataFrame({"feat": ["f1", "f2", "f3"], "subset1": [0, 1, 0]})
        with self.assertRaises(ValueError):
            self.fsp.check_feature_subset_file(feature_specs, "subset1")

    def test_check_feature_subset_file_no_subset_column(self):
        feature_specs = pd.DataFrame({"Feature": ["f1", "f2", "f3"], "subset1": [0, 1, 0]})
        with self.assertRaises(ValueError):
            self.fsp.check_feature_subset_file(feature_specs, "subset2")

    def test_check_feature_subset_file_wrong_values_in_subset(self):
        feature_specs = pd.DataFrame(
            {"Feature": ["f1", "f2", "f3"], "subset1": ["yes", "no", "yes"]}
        )
        with self.assertRaises(ValueError):
            self.fsp.check_feature_subset_file(feature_specs, "subset1")

    def test_check_feature_subset_file_no_sign_column(self):
        feature_specs = pd.DataFrame({"feature": ["f1", "f2", "f3"], "subset1": [0, 1, 0]})
        with self.assertRaises(ValueError):
            self.fsp.check_feature_subset_file(feature_specs, sign="subset1")

    def test_check_feature_subset_file_wrong_values_in_sign(self):
        feature_specs = pd.DataFrame(
            {"Feature": ["f1", "f2", "f3"], "sign_SYS1": ["+1", "-1", "+1"]}
        )
        with self.assertRaises(ValueError):
            self.fsp.check_feature_subset_file(feature_specs, sign="SYS1")
