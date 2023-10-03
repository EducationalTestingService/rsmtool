"""
Class for writing DataContainer frames to disk.

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""
from os import makedirs
from os.path import join

from .utils.wandb import log_dataframe_to_wandb


class DataWriter:
    """Class to write out DataContainer objects."""

    def __init__(self, experiment_id=None, context=None, wandb_run=None):
        """
        Initialize the DataWriter object.

        Parameters
        ----------
        experiment_id : str
            The experiment name to be used in the output file names
        context : str
            The context in which this writer is used. Defaults to ``None``.
        wandb_run : wandb.Run
            The wandb run object if wandb is enabled, None otherwise.
            If enabled, all the output data frames will be logged to
            this run as tables.
            Defaults to ``None``.
        """
        self._id = experiment_id
        self.context = context
        self.wandb_run = wandb_run

    @staticmethod
    def write_frame_to_file(df, name_prefix, file_format="csv", index=False, **kwargs):
        """
        Write given data frame to disk with given name and file format.

        Parameters
        ----------
        df : pandas DataFrame
            Data frame to write to disk
        name_prefix : str
            The complete prefix for the file to be written to disk.
            This includes everything except the extension.
        file_format : str
            The file format (extension) for the file to be written to disk.
            One of {"csv", "xlsx", "tsv"}.
            Defaults to "csv".
        index : bool, optional
            Whether to include the index in the output file.
            Defaults to ``False``.

        Raises
        ------
        KeyError
            If ``file_format`` is not valid.
        """
        file_format = file_format.lower()

        if file_format == "csv":
            name_prefix += ".csv"
            df.to_csv(name_prefix, index=index, **kwargs)

        elif file_format == "tsv":
            name_prefix += ".tsv"
            df.to_csv(name_prefix, index=index, sep="\t", **kwargs)

        # Added jsonlines for experimental purposes, but leaving
        # this out of the documentation at this stage
        elif file_format == "jsonlines":
            name_prefix += ".jsonlines"
            df.to_json(name_prefix, orient="records", lines=True, **kwargs)

        elif file_format == "xlsx":
            name_prefix += ".xlsx"
            df.to_excel(name_prefix, index=index, **kwargs)

        else:
            raise KeyError(
                "Please make sure that the `file_format` specified "
                "is one of the following:\n{`csv`, `tsv`, `xlsx`}.\n"
                f"You specified {file_format}."
            )

    def write_experiment_output(
        self,
        csvdir,
        container_or_dict,
        dataframe_names=None,
        new_names_dict=None,
        include_experiment_id=True,
        reset_index=False,
        file_format="csv",
        index=False,
        **kwargs,
    ):
        """
        Write out each of the named frames to disk.

        This function writes out each of the given list of data frames as a
        ".csv", ".tsv", or ``.xlsx`` file in the given directory. Each data
        frame was generated as part of running an RSMTool experiment. All files
        are prefixed with the given experiment ID and suffixed with either the
        name of the data frame in the DataContainer (or dict) object, or a new
        name if ``new_names_dict`` is specified. Additionally, the indexes in
        the data frames are reset if so specified.

        Parameters
        ----------
        csvdir : str
            Path to the output experiment sub-directory that will
            contain the CSV files corresponding to each of the data frames.
        container_or_dict : container.DataContainer or dict
            A DataContainer object or dict, where keys are data frame
            names and values are ``pd.DataFrame`` objects.
        dataframe_names : list of str, optional
            List of data frame names, one for each of the data frames.
            Defaults to ``None``.
        new_names_dict : dict, optional
            New dictionary with new names for the data frames, if desired.
            Defaults to ``None``.
        include_experiment_id : str, optional
            Whether to include the experiment ID in the file name.
            Defaults to ``True``.
        reset_index : bool, optional
            Whether to reset the index of each data frame
            before writing to disk.
            Defaults to ``False``.
        file_format : str, optional
            The file format in which to output the data.
            One of {"csv", "xlsx", "tsv"}.
            Defaults to "csv".
        index : bool, optional
            Whether to include the index in the output file.
            Defaults to ``False``.


        Raises
        ------
        KeyError
            If ``file_format`` is not valid, or a named data frame
            is not present in ``container_or_dict``.
        """
        container_or_dict = container_or_dict.copy()

        # If no `dataframe_names` specified, use all names
        if dataframe_names is None:
            dataframe_names = container_or_dict.keys()

        # Otherwise, check to make sure all specified names
        # are actually in the DataContainer
        else:
            for name in dataframe_names:
                if name not in container_or_dict:
                    raise KeyError(f"The name `{name}` is not in the container or dictionary.")

        # Loop through DataFrames, and save
        # output in specified format
        for dataframe_name in dataframe_names:
            df = container_or_dict[dataframe_name]
            if df is None:
                raise KeyError(f"The DataFrame `{dataframe_name}` does not exist.")

            # If the DataFrame is empty, skip it
            if df.empty:
                continue

            # If there is a desire to rename the DataFrame,
            # get the new name
            if new_names_dict is not None:
                if dataframe_name in new_names_dict:
                    dataframe_name = new_names_dict[dataframe_name]

            # Reset the index, if desired
            if reset_index:
                df.index.name = ""
                df.reset_index(inplace=True)

            # If include_experiment_id is True, and the experiment_id exists
            # include it in the file name; otherwise, do not include it.
            if include_experiment_id and self._id is not None:
                outfile = join(csvdir, f"{self._id}_{dataframe_name}")
            else:
                outfile = join(csvdir, dataframe_name)

            # write out the frame to disk in the given file
            self.write_frame_to_file(df, outfile, file_format=file_format, index=index, **kwargs)
            log_dataframe_to_wandb(self.wandb_run, df, dataframe_name, self.context)

    def write_feature_csv(
        self,
        featuredir,
        data_container,
        selected_features,
        include_experiment_id=True,
        file_format="csv",
    ):
        """
        Write out the selected features to disk.

        Parameters
        ----------
        featuredir : str
            Path to the experiment output directory where the
            feature JSON file will be saved.
        data_container : container.DataContainer
            A data container object.
        selected_features : list of str
            List of features that were selected for model building.
        include_experiment_id : bool, optional
            Whether to include the experiment ID in the file name.
            Defaults to ``True``.
        file_format : str, optional
            The file format in which to output the data.
            One of {"csv", "tsv", "xlsx"}.
            Defaults to "csv".
        """
        df_feature_specs = data_container["feature_specs"]

        # Select specific features used in training
        df_selected = df_feature_specs[df_feature_specs["feature"].isin(selected_features)]

        # Replace existing `feature_specs` with selected features specs
        data_container.add_dataset({"frame": df_selected, "name": "feature_specs"}, update=True)

        makedirs(featuredir, exist_ok=True)
        self.write_experiment_output(
            featuredir,
            data_container,
            ["feature_specs"],
            {"feature_specs": "selected"},
            include_experiment_id=include_experiment_id,
            file_format=file_format,
        )
