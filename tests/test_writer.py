import os
import unittest
from itertools import product
from pathlib import Path
from shutil import rmtree
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from rsmtool.container import DataContainer
from rsmtool.writer import DataWriter


class TestDataWriter(unittest.TestCase):
    def check_write_frame_to_file(self, file_format, include_index):
        # create a dummy data frame for testing
        df_to_write = pd.DataFrame(np.random.normal(size=(120, 3)), columns=["A", "B", "C"])

        # create a temporary directory where the file will be written
        tempdir = TemporaryDirectory()

        # create a name prefix for the frame
        name_prefix = Path(tempdir.name) / "test_frame"

        # write the frame to disk
        DataWriter.write_frame_to_file(
            df_to_write, str(name_prefix), file_format=file_format, index=include_index
        )

        # check that the file was written to disk
        dir_listing = os.listdir(tempdir.name)
        self.assertTrue(f"test_frame.{file_format}" in dir_listing)

        # read the file and check that the frame was written as expected
        if file_format == "csv":
            df_written = pd.read_csv(f"{name_prefix}.{file_format}")
        elif file_format == "tsv":
            df_written = pd.read_csv(f"{name_prefix}.{file_format}", sep="\t")
        elif file_format == "xlsx":
            df_written = pd.read_excel(f"{name_prefix}.{file_format}")
        else:
            df_written = pd.read_json(f"{name_prefix}.{file_format}", orient="records", lines=True)

        # check that the index is there if it's supposed to be
        if include_index and file_format in ["csv", "tsv", "xlsx"]:
            self.assertTrue("Unnamed: 0" in df_written.columns)

        # check that the data is the same
        df_written = df_written[["A", "B", "C"]]
        assert_frame_equal(df_to_write, df_written)

        # clean up the temporary directory
        tempdir.cleanup()

    def test_write_frame_to_file(self):
        for file_format, include_index in product(
            ["csv", "tsv", "xlsx", "jsonlines"], [False, True]
        ):
            yield self.check_write_frame_to_file, file_format, include_index

    def test_data_container_save_files(self):
        data_sets = [
            {
                "name": "dataset1",
                "frame": pd.DataFrame(np.random.normal(size=(100, 2)), columns=["A", "B"]),
            },
            {
                "name": "dataset2",
                "frame": pd.DataFrame(np.random.normal(size=(120, 3)), columns=["A", "B", "C"]),
            },
        ]

        container = DataContainer(data_sets)

        directory = "temp_directory_data_container_save_files_xyz"
        os.makedirs(directory, exist_ok=True)

        writer = DataWriter()
        for file_type in ["jsonlines", "csv", "xlsx"]:
            if file_type != "jsonlines":
                writer.write_experiment_output(
                    directory,
                    container,
                    dataframe_names=["dataset1"],
                    file_format=file_type,
                )
            else:
                writer.write_experiment_output(
                    directory,
                    container,
                    new_names_dict={"dataset1": "aaa"},
                    dataframe_names=["dataset1"],
                    file_format=file_type,
                )

        aaa_json = pd.read_json(
            os.path.join(directory, "aaa.jsonlines"), orient="records", lines=True
        )
        ds_1_csv = pd.read_csv(os.path.join(directory, "dataset1.csv"))
        ds_1_xls = pd.read_excel(os.path.join(directory, "dataset1.xlsx"))

        output_dir = os.listdir(directory)
        rmtree(directory)
        self.assertEqual(
            sorted(output_dir), sorted(["aaa.jsonlines", "dataset1.csv", "dataset1.xlsx"])
        )

        assert_frame_equal(container.dataset1, aaa_json)
        assert_frame_equal(container.dataset1, ds_1_csv)
        assert_frame_equal(container.dataset1, ds_1_xls)

    def test_dictionary_save_files(self):
        data_sets = {
            "dataset1": pd.DataFrame(np.random.normal(size=(100, 2)), columns=["A", "B"]),
            "dataset2": pd.DataFrame(np.random.normal(size=(120, 3)), columns=["A", "B", "C"]),
        }

        directory = "temp_directory_dictionary_save_files_xyz"
        os.makedirs(directory, exist_ok=True)

        writer = DataWriter()
        for file_type in ["jsonlines", "csv", "xlsx"]:
            if file_type != "jsonlines":
                writer.write_experiment_output(
                    directory,
                    data_sets,
                    dataframe_names=["dataset1"],
                    file_format=file_type,
                )
            else:
                writer.write_experiment_output(
                    directory,
                    data_sets,
                    new_names_dict={"dataset1": "aaa"},
                    dataframe_names=["dataset1"],
                    file_format=file_type,
                )

        aaa_json = pd.read_json(
            os.path.join(directory, "aaa.jsonlines"), orient="records", lines=True
        )
        ds_1_csv = pd.read_csv(os.path.join(directory, "dataset1.csv"))
        ds_1_xls = pd.read_excel(os.path.join(directory, "dataset1.xlsx"))

        output_dir = os.listdir(directory)
        rmtree(directory)
        self.assertEqual(
            sorted(output_dir), sorted(["aaa.jsonlines", "dataset1.csv", "dataset1.xlsx"])
        )

        assert_frame_equal(data_sets["dataset1"], aaa_json)
        assert_frame_equal(data_sets["dataset1"], ds_1_csv)
        assert_frame_equal(data_sets["dataset1"], ds_1_xls)

    def test_data_container_save_wrong_format(self):
        data_sets = [
            {
                "name": "dataset1",
                "frame": pd.DataFrame(np.random.normal(size=(100, 2)), columns=["A", "B"]),
            },
            {
                "name": "dataset2",
                "frame": pd.DataFrame(np.random.normal(size=(120, 3)), columns=["A", "B", "C"]),
            },
        ]

        container = DataContainer(data_sets)

        directory = "temp_directory_container_save_wrong_format_xyz"

        writer = DataWriter()
        with self.assertRaises(KeyError):
            writer.write_experiment_output(
                directory, container, dataframe_names=["dataset1"], file_format="html"
            )

    def test_data_container_save_files_with_id(self):
        data_sets = [
            {
                "name": "dataset1",
                "frame": pd.DataFrame(np.random.normal(size=(100, 2)), columns=["A", "B"]),
            },
            {
                "name": "dataset2",
                "frame": pd.DataFrame(np.random.normal(size=(120, 3)), columns=["A", "B", "C"]),
            },
        ]

        container = DataContainer(data_sets)

        directory = "temp_directory_save_files_with_id_xyz"
        os.makedirs(directory, exist_ok=True)

        writer = DataWriter("test")
        for file_type in ["jsonlines", "csv", "xlsx"]:
            if file_type != "jsonlines":
                writer.write_experiment_output(
                    directory,
                    container,
                    dataframe_names=["dataset1"],
                    file_format=file_type,
                )
            else:
                writer.write_experiment_output(
                    directory,
                    container,
                    new_names_dict={"dataset1": "aaa"},
                    dataframe_names=["dataset1"],
                    file_format=file_type,
                )

        aaa_json = pd.read_json(
            os.path.join(directory, "test_aaa.jsonlines"), orient="records", lines=True
        )
        ds_1_csv = pd.read_csv(os.path.join(directory, "test_dataset1.csv"))
        ds_1_xls = pd.read_excel(os.path.join(directory, "test_dataset1.xlsx"))

        output_dir = os.listdir(directory)
        rmtree(directory)
        self.assertEqual(
            sorted(output_dir),
            sorted(["test_aaa.jsonlines", "test_dataset1.csv", "test_dataset1.xlsx"]),
        )

        assert_frame_equal(container.dataset1, aaa_json)
        assert_frame_equal(container.dataset1, ds_1_csv)
        assert_frame_equal(container.dataset1, ds_1_xls)
