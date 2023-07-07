import json
import os
import tempfile
import unittest
import warnings
from shutil import rmtree

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from rsmtool.reader import DataReader, read_jsonlines, try_to_load_file


class TestReader(unittest.TestCase):
    def test_try_to_load_file_none(self):
        assert try_to_load_file("bdadui88asldfkas;j.sarasd") is None

    def test_try_to_load_file_fail(self):
        with self.assertRaises(FileNotFoundError):
            try_to_load_file("bdadui88asldfkas;j.sarasd", raise_error=True)

    def test_try_to_load_file_warn(self):
        with self.assertRaises(Warning):
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                try_to_load_file("bdadui88asldfkas;j.sarasd", raise_warning=True)


class TestDataReader(unittest.TestCase):
    filepaths = []
    df_train = pd.DataFrame(
        {
            "id": ["001", "002", "003"],
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
            "gender": ["M", "F", "F"],
            "candidate": ["123", "456", "78901"],
        }
    )

    df_test = pd.DataFrame(
        {
            "id": ["102", "102", "103"],
            "feature1": [5, 3, 2],
            "feature2": [3, 4, 3],
            "gender": ["F", "M", "F"],
            "candidate": ["135", "546", "781"],
        }
    )

    df_specs = pd.DataFrame(
        {
            "feature": ["f1", "f2", "f3"],
            "transform": ["raw", "inv", "sqrt"],
            "sign": ["+", "+", "-"],
        }
    )

    df_other = pd.DataFrame({"random": ["a", "b", "c"], "things": [1241, 45332, 3252]})

    @classmethod
    def tearDownClass(cls):
        for path in cls.filepaths:
            if os.path.exists(path):
                os.unlink(path)
        cls.filepaths = []

    @staticmethod
    def make_file_from_ext(df, ext):
        tempf = tempfile.NamedTemporaryFile(mode="w", suffix=f".{ext}", delete=False)
        if ext.lower() == "csv":
            df.to_csv(tempf.name, index=False)
        elif ext.lower() == "tsv":
            df.to_csv(tempf.name, sep="\t", index=False)
        elif ext.lower() == "xlsx":
            df.to_excel(tempf.name, index=False)
        elif ext.lower() in ["jsonlines"]:
            df.to_json(tempf.name, orient="records", lines=True)
        tempf.close()
        return tempf.name

    def get_container(self, name_ext_tuples, converters=None):
        """Get DataContainer object from a list of (name, ext) tuples."""
        names_ = []
        paths_ = []
        for name, ext in name_ext_tuples:
            if name == "train":
                df = self.df_train
            elif name == "test":
                df = self.df_test
            elif name == "feature_specs":
                df = self.df_specs
            else:
                df = self.df_other

            path = TestDataReader.make_file_from_ext(df, ext)

            names_.append(name)
            paths_.append(path)

        reader = DataReader(paths_, names_, converters)
        container = reader.read()

        self.filepaths.extend(paths_)
        return container

    def check_read_from_file(self, extension):
        """Test whether ``read_from_file()`` works as expected."""
        name = TestDataReader.make_file_from_ext(self.df_train, extension)

        # now read in the file using `read_data_file()`
        df_read = DataReader.read_from_file(name, converters={"id": str, "candidate": str})

        # Make sure we get rid of the file at the end,
        # at least if we get to this point (i.e. no errors raised)
        self.filepaths.append(name)

        assert_frame_equal(self.df_train, df_read)

    def check_train(self, name_ext_tuples, converters=None):
        container = self.get_container(name_ext_tuples, converters)
        frame = container.train
        assert_frame_equal(frame.sort_index(axis=1), self.df_train.sort_index(axis=1))

    def check_feature_specs(self, name_ext_tuples, converters=None):
        container = self.get_container(name_ext_tuples, converters)
        frame = container.feature_specs
        assert_frame_equal(frame.sort_index(axis=1), self.df_specs.sort_index(axis=1))

    def test_read_data_file(self):
        # note that we cannot check for capital .xls and .xlsx
        # because xlwt does not support these extensions
        for extension in ["csv", "tsv", "xlsx", "CSV", "TSV"]:
            yield self.check_read_from_file, extension

    def test_read_data_file_wrong_extension(self):
        with self.assertRaises(ValueError):
            self.check_read_from_file("txt")

    def test_container_train_property(self):
        test_lists = [
            [("train", "csv"), ("test", "tsv")],
            [("train", "csv"), ("feature_specs", "xlsx")],
            [("train", "csv"), ("test", "xlsx"), ("train_metadata", "tsv")],
            [("train", "jsonlines"), ("test", "jsonlines")],
        ]

        converter = {
            "id": str,
            "feature1": np.int64,
            "feature2": np.int64,
            "candidate": str,
        }
        converters = [
            {"train": converter, "test": converter},
            {"train": converter},
            {"train": converter, "test": converter},
            {"train": converter, "test": converter},
        ]
        for idx, test_list in enumerate(test_lists):
            yield self.check_train, test_list, converters[idx]

    def test_container_feature_specs_property(self):
        test_lists = [
            [("feature_specs", "csv"), ("test", "tsv")],
            [("train", "csv"), ("feature_specs", "xlsx")],
            [("train", "csv"), ("feature_specs", "tsv"), ("train_metadata", "tsv")],
            [("train", "jsonlines"), ("feature_specs", "jsonlines")],
        ]
        for test_list in test_lists:
            yield self.check_feature_specs, test_list

    def test_no_container_feature_specs_property(self):
        test_lists = [("train", "csv"), ("test", "tsv"), ("train_metadata", "xlsx")]
        container = self.get_container(test_lists)
        with self.assertRaises(AttributeError):
            container.feature_specs

    def test_no_container_test_property(self):
        test_lists = [
            ("feature_specs", "csv"),
            ("train", "tsv"),
            ("train_metadata", "xlsx"),
        ]
        container = self.get_container(test_lists)
        with self.assertRaises(AttributeError):
            container.test

    def test_container_test_property_frame_equal(self):
        test_lists = [
            ("feature_specs", "csv"),
            ("test", "tsv"),
            ("train_metadata", "xlsx"),
        ]
        converter = {
            "id": str,
            "feature1": np.int64,
            "feature2": np.int64,
            "candidate": str,
        }
        converters = {"test": converter}
        container = self.get_container(test_lists, converters)
        frame = container.test
        assert_frame_equal(frame, self.df_test)

    def test_get_values(self):
        test_lists = [("feature_specs", "csv")]
        container = self.get_container(test_lists)
        frame = container["feature_specs"]
        assert_frame_equal(container.values()[0], frame)

    def test_length(self):
        test_lists = [("feature_specs", "csv")]
        container = self.get_container(test_lists)
        self.assertEqual(len(container), 1)

    def test_get_path_default(self):
        test_lists = [("feature_specs", "csv")]
        container = self.get_container(test_lists)
        self.assertEqual(container.get_path("aaa"), None)

    def test_getitem_test_from_key(self):
        test_lists = [("feature_specs", "csv"), ("test", "tsv"), ("train", "xlsx")]
        converter = {
            "id": str,
            "feature1": np.int64,
            "feature2": np.int64,
            "candidate": str,
        }
        converters = {"train": converter, "test": converter}
        container = self.get_container(test_lists, converters)
        frame = container["test"]
        assert_frame_equal(frame, self.df_test)

    def test_add_containers(self):
        test_list1 = [("feature_specs", "csv"), ("train", "xlsx")]
        container1 = self.get_container(test_list1)

        test_list2 = [("test", "csv"), ("train_metadata", "tsv")]
        container2 = self.get_container(test_list2)

        container3 = container1 + container2
        names = sorted(container3.keys())
        self.assertEqual(names, ["feature_specs", "test", "train", "train_metadata"])

    def test_add_containers_duplicate_keys(self):
        test_list1 = [("feature_specs", "csv"), ("train", "xlsx")]
        container1 = self.get_container(test_list1)

        test_list2 = [("test", "csv"), ("train", "tsv")]
        container2 = self.get_container(test_list2)
        with self.assertRaises(KeyError):
            container1 + container2

    def test_locate_files_list(self):
        paths = ["file1.csv", "file2.xlsx"]
        config_dir = "output"
        result = DataReader.locate_files(paths, config_dir)
        assert isinstance(result, list)
        self.assertEqual(result, [None, None])

    def test_locate_files_str(self):
        paths = "file1.csv"
        config_dir = "output"
        result = DataReader.locate_files(paths, config_dir)
        self.assertEqual(result, None)

    def test_locate_files_works(self):
        config_dir = "temp_output"
        os.makedirs(config_dir, exist_ok=True)

        paths = "file1.csv"
        full_path = os.path.abspath(os.path.join(config_dir, paths))
        open(full_path, "a").close()

        result = DataReader.locate_files(paths, config_dir)
        rmtree(config_dir)
        self.assertEqual(result, full_path)

    def test_locate_files_wrong_type(self):
        paths = {"file1.csv", "file2.xlsx"}
        config_dir = "output"
        with self.assertRaises(ValueError):
            DataReader.locate_files(paths, config_dir)

    def test_setup_none_in_path(self):
        paths = ["path1.csv", None, "path2.csv"]
        framenames = ["train", "test", "features"]
        with self.assertRaises(ValueError):
            DataReader(paths, framenames)


class TestJsonLines(unittest.TestCase):
    def setUp(self):
        self.filepaths = []

        self.expected = pd.DataFrame(
            {
                "id": ["001", "002", "003"],
                "feature1": [1, 2, 3],
                "feature2": [1.5, 2.5, 3.5],
            }
        )

    def tearDown(self):
        for path in self.filepaths:
            if os.path.exists(path):
                os.unlink(path)
        self.filepaths = []

    @staticmethod
    def create_jsonlines_file(jsondict):
        tempf = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonlines", delete=False)
        for entry in jsondict:
            json.dump(entry, tempf)
            tempf.write("\n")
        return tempf.name

    def check_jsonlines_output(self, jsondict):
        fname = self.create_jsonlines_file(jsondict)
        self.filepaths.append(fname)
        df = read_jsonlines(fname, converters={"id": str})
        assert_frame_equal(df.sort_index(axis=1), self.expected.sort_index(axis=1))

    def test_read_jsonlines(self):
        jsonlines = [
            {"id": "001", "feature1": 1, "feature2": 1.5},
            {"id": "002", "feature1": 2, "feature2": 2.5},
            {"id": "003", "feature1": 3, "feature2": 3.5},
        ]
        self.check_jsonlines_output(jsonlines)

    def test_read_nested_jsonlines(self):
        nested_jsonlines = [
            {"id": "001", "features": {"feature1": 1, "feature2": 1.5}},
            {"id": "002", "features": {"feature1": 2, "feature2": 2.5}},
            {"id": "003", "features": {"feature1": 3, "feature2": 3.5}},
        ]
        self.check_jsonlines_output(nested_jsonlines)

    def test_read_nested_jsonlines_all_nested(self):
        all_nested_jsonlines = [
            {"values": {"id": "001", "feature1": 1, "feature2": 1.5}},
            {"values": {"id": "002", "feature1": 2, "feature2": 2.5}},
            {"values": {"id": "003", "feature1": 3, "feature2": 3.5}},
        ]
        self.check_jsonlines_output(all_nested_jsonlines)

    def test_read_jsonlines_more_than_2_levels(self):
        multi_nested_jsonlines = [
            {"values": {"id": "001", "features": {"feature1": 1, "feature2": 1.5}}},
            {"values": {"id": "002", "features": {"feature1": 2, "feature2": 2.5}}},
            {"values": {"id": "003", "features": {"feature1": 3, "feature2": 3.5}}},
        ]
        self.expected.columns = ["id", "features.feature1", "features.feature2"]
        self.check_jsonlines_output(multi_nested_jsonlines)

    def test_read_jsonlines_single_line(self):
        jsonlines = [{"id": "001", "feature1": 1, "feature2": 1.5}]
        self.expected = self.expected.iloc[0:1]
        self.check_jsonlines_output(jsonlines)

    def test_read_jsonlines_mismatched_keys(self):
        all_nested_jsonlines = [
            {"values": {"id": "001", "feature1": 1, "feature2": 1.5}},
            {"values": {"id": "002", "feature2": 2, "feature3": 2.5}},
            {"values": {"id": "003", "feature1": 3}},
        ]
        self.expected = pd.DataFrame(
            {
                "id": self.expected["id"],
                "feature1": [1, np.nan, 3],
                "feature2": [1.5, 2, np.nan],
                "feature3": [np.nan, 2.5, np.nan],
            }
        )
        self.check_jsonlines_output(all_nested_jsonlines)

    def test_read_jsons_with_nulls(self):
        """Test if null in JSONs are properly read as ``None``."""
        all_nested_jsonlines = [
            {"values": {"id": "001", "feature1": None, "feature2": 1.5}},
            {"values": {"id": "002", "feature1": 2, "feature2": None}},
            {"values": {"id": "003", "feature1": 3, "feature2": None}},
        ]
        self.expected.loc[0, "feature1"] = np.nan
        self.expected.loc[1, "feature2"] = np.nan
        self.expected.loc[2, "feature2"] = np.nan
        self.check_jsonlines_output(all_nested_jsonlines)

    def test_read_json_with_NaNs(self):
        #####################################################
        # NOTE: This test is no longer failing              #
        # because Pandas can now parse NaNs:                #
        # https://github.com/pandas-dev/pandas/issues/12213 #
        #####################################################
        all_nested_jsonlines = [
            {"values": {"id": "001", "feature1": np.nan, "feature2": 1.5}},
            {"values": {"id": "002", "feature1": 2, "feature2": np.nan}},
            {"values": {"id": "003", "feature1": 3, "feature2": np.nan}},
        ]
        self.expected.loc[0, "feature1"] = np.nan
        self.expected.loc[1, "feature2"] = np.nan
        self.expected.loc[2, "feature2"] = np.nan
        self.check_jsonlines_output(all_nested_jsonlines)

    def test_read_plain_json(self):
        plain_json = {
            "id": ["001", "002", "003"],
            "feature1": [1, 2, 3],
            "feature2": [1.5, 2.5, 3.5],
        }
        with self.assertRaises(ValueError):
            self.check_jsonlines_output(plain_json)
