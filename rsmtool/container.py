"""
Class to encapsulate data contained in multiple pandas DataFrames.

It represents each of the multiple data sources as a "dataset". Each
dataset is represented by three properties:
- "name" : the name of the data set
- "frame" : the pandas DataFrame that contains the actual data
- "path" : the path to the file on disk from which the data was read

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

import warnings
from copy import copy, deepcopy


class DataContainer:
    """Class to encapsulate datasets."""

    def __init__(self, datasets=None):
        """
        Initialize a DataContainer object.

        Parameters
        ----------
        datasets : list of dicts, optional
            A list of dataset dictionaries. Each dict should have the
            following keys: "name" containing the name of the dataset,
            "frame" containing the dataframe object representing the
            dataset, and "path" containing the path to the file from
            which the frame was read.
        """
        self._names = []
        self._dataframes = {}
        self._data_paths = {}

        if datasets is not None:
            for dataset_dict in datasets:
                self.add_dataset(dataset_dict, update=False)

    def __contains__(self, name):
        """
        Check if the container object contains a dataset with a given name.

        Parameters
        ----------
        name : str
            The name to check in the container object.

        Returns
        -------
        key_check : bool
            ``True`` if a dataset with this name exists in the container
            object, else ``False``.
        """
        return name in self._names

    def __getitem__(self, name):
        """
        Get the data frame for the dataset with the given name.

        Parameters
        ----------
        name : str
            The name for the dataset.

        Returns
        -------
        frame : pandas DataFrame
            The data frame for the dataset with the given name.

        Raises
        ------
        KeyError
            If the name does not exist in the container.
        """
        return self.get_frame(name)

    def __len__(self):
        """
        Return the number of datasets in the container.

        Returns
        -------
        length : int
            The size of the container (i.e. number of datasets).
        """
        return len(self._names)

    def __str__(self):
        """
        Return a string representation of the container.

        Returns
        -------
        container_names : str
            A comma-separated list of dataset names from the container.
        """
        return ", ".join(self._names)

    def __add__(self, other):
        """
        Add another container object to instance.

        Return a new container object with datasets included
        in either of the two containers.

        Parameters
        ----------
        other : DataContainer
            The container object to add.

        Returns
        -------
        output : DataContainer
            New container object containing datasets
            included in this instance and the other instance.

        Raises
        ------
        KeyError
            If there are duplicate keys in the two containers.
        ValueError
            If the object being added is not a container.

        """
        if not isinstance(other, DataContainer):
            raise ValueError(f"Object must be a `DataContainer`, not {type(other)}.")

        # Make sure there are no duplicate keys
        common_keys = set(other._names).intersection(self._names)
        if common_keys:
            raise KeyError(f"The key(s) `{', '.join(common_keys)}` already exist in the container.")

        dicts = DataContainer.to_datasets(self)
        dicts.extend(DataContainer.to_datasets(other))
        return DataContainer(dicts)

    def __iter__(self):
        """
        Iterate through the container keys (dataset names).

        Yields
        ------
        key
            A key (name) in the container dictionary.
        """
        for key in self.keys():
            yield key

    @staticmethod
    def to_datasets(data_container):
        """
        Convert container object to a list of dataset dictionaries.

        Each dictionary will contain the "name", "frame", and
        "path" keys.

        Parameters
        ----------
        data_container : DataContainer
            The container object to convert.

        Returns
        -------
        datasets_dict : list of dicts
            A list of dataset dictionaries.
        """
        dataset_dicts = []
        for name in data_container.keys():
            dataset_dict = {
                "name": name,
                "path": data_container.get_path(name),
                "frame": data_container.get_frame(name),
            }
            dataset_dicts.append(dataset_dict)
        return dataset_dicts

    def add_dataset(self, dataset_dict, update=False):
        """
        Add a new dataset (or update an existing one).

        Parameters
        ----------
        dataset_dict : dict
            The dataset dictionary to add or update
            with the "name", "frame", and "path" keys.
        update : bool, optional
            Update an existing DataFrame, if ``True``.
            Defaults to ``False``.
        """
        name = dataset_dict["name"]
        data_frame = dataset_dict["frame"]
        path = dataset_dict.get("path")

        if not update:
            if name in self._names:
                raise KeyError(f"The name {name} already exists in the container dictionary.")

        if name not in self._names:
            self._names.append(name)

        self._dataframes[name] = data_frame
        self._data_paths[name] = path

        self.__setattr__(name, data_frame)

    def get_path(self, name, default=None):
        """
        Get the path for the dataset given the name.

        Parameters
        ----------
        name : str
            The name for the dataset.
        default : str, optional
            The default path to return if the named dataset does not exist.
            Defaults to ``None``.

        Returns
        -------
        path : str
            The path for the named dataset.
        """
        if name not in self._names:
            return default
        return self._data_paths[name]

    def get_frame(self, name, default=None):
        """
        Get the data frame given the dataset name.

        Parameters
        ----------
        name : str
            The name for the dataset.
        default : pandas DataFrame, optional
            The default value to return if the named dataset does not exist.
            Defaults to ``None``.

        Returns
        -------
        frame : pandas DataFrame
            The data frame for the named dataset.
        """
        if name not in self._names:
            return default
        return self._dataframes[name]

    def get_frames(self, prefix=None, suffix=None):
        """
        Get all data frames with a given prefix or suffix in their name.

        Note that the selection by prefix or suffix is case-insensitive.

        Parameters
        ----------
        prefix : str, optional
            Only return frames with the given prefix. If ``None``, then
            do not exclude any frames based on their prefix.
            Defaults to ``None``.
        suffix : str, optional
            Only return frames with the given suffix. If ``None``, then
            do not exclude any frames based on their suffix.
            Defaults to ``None``.

        Returns
        -------
        frames : dict
            A dictionary with the data frames that contain the specified
            prefix and/or suffix in their corresponding names. The names
            are the keys and the frames are the values.
        """
        if prefix is None:
            prefix = ""

        if suffix is None:
            suffix = ""

        names = [
            name
            for name in self._names
            if name.lower().startswith(prefix) and name.lower().endswith(suffix)
        ]

        frames = {}
        for name in names:
            frames[name] = self._dataframes[name]
        return frames

    def keys(self):  # noqa: D402
        """
        Return the container keys (dataset names) as a list.

        Returns
        -------
        keys : list
            A list of keys (names) in the container object.
        """
        return self._names

    def values(self):
        """
        Return all data frames as a list.

        Returns
        -------
        values : list
            A list of all data frames in the container object.
        """
        return [self._dataframes[name] for name in self._names]

    def items(self):
        """
        Return the container items as a list of (name, frame) tuples.

        Returns
        -------
        items : list of tuples
            A list of (name, frame) tuples in the container object.
        """
        return [(name, self._dataframe[name]) for name in self._names]

    def drop(self, name):
        """
        Drop a given dataset from the container and return instance.

        Parameters
        ----------
        name : str
            The name of the dataset to drop.

        Returns
        -------
        self
        """
        if name not in self:
            warnings.warn(
                f"The name `{name}` is not in the container. " f"No datasets will be dropped."
            )
        else:
            self._names.remove(name)
            self._dataframes.pop(name)
            self._data_paths.pop(name)
        return self

    def rename(self, name, new_name):
        """
        Rename a given dataset in the container and return instance.

        Parameters
        ----------
        name : str
            The name of the current dataset in the container object.
        new_name : str
            The new name for the dataset in the container object.

        Returns
        -------
        self
        """
        if name not in self:
            warnings.warn(f"The name `{name}` is not in the container and cannot be renamed.")
        else:
            frame = self._dataframes[name]
            path = self._data_paths[name]
            self.add_dataset({"name": new_name, "frame": frame, "path": path}, update=True)
            self.drop(name)
        return self

    def copy(self, deep=True):
        """
        Return a copy of the container object.

        Parameters
        ----------
        deep : bool, optional
            If ``True``, create a deep copy of the underlying data frames.
            Defaults to ``True``.
        """
        if deep:
            return deepcopy(self)
        return copy(self)
