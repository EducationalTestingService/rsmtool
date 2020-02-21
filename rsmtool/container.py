"""
Classes for storing any kind of data contained
in a pd.DataFrame object.

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

import warnings
from copy import copy, deepcopy


class DataContainer:
    """
    A class to encapsulate datasets.
    """

    def __init__(self,
                 datasets=None):
        """
        Initialize ``DataContainer`` object.

        Parameters
        ----------
        datasets : list of dicts, optional
            A list of dataset dicts. Each dict should have the following keys:
            ``name`` containing the name of the dataset, ``frame`` containing
            the dataframe object that contains the dataset, and ``path`` containing
            the file from which the dataset was read.
        """
        self._names = []
        self._dataframes = {}
        self._data_paths = {}

        if datasets is not None:
            for dataset_dict in datasets:
                self.add_dataset(dataset_dict,
                                 update=False)

    def __contains__(self, key):
        """
        Check if DataContainer object
        contains a given key.

        Parameters
        ----------
        key : str
            A key to check in the DataContainer object

        Returns
        -------
        key_check : bool
            True if key in DataContainer object, else False
        """
        return key in self._names

    def __getitem__(self, key):
        """
        Get frame, given key.

        Parameters
        ----------
        key : str
            Name for the data.

        Returns
        -------
        frame : pd.DataFrame
            The DataFrame.

        Raises
        ------
        KeyError
            If the key does not exist.
        """
        return self.get_frame(key)

    def __len__(self):
        """
        Return the length of the
        DataContainer names.

        Returns
        -------
        length : int
            The length of the container (i.e. number of frames)
        """
        return len(self._names)

    def __str__(self):
        """
        String representation of the object.

        Returns
        -------
        container_names : str
            A comma-separated list of names from the container.
        """
        return ', '.join(self._names)

    def __add__(self, other):
        """
        Add two DataContainer objects together and return a new
        DataContainer object with DataFrames common to both
        DataContainers

        Raises
        ------
        ValueError
            If the object being added is not a DataContainer
        KeyError
            If there are duplicate keys in the two DataContainers.
        """
        if not isinstance(other, DataContainer):
            raise ValueError('Object must be `DataContainer`, '
                             'not {}.'.format(type(other)))

        # Make sure there are no duplicate keys
        common_keys = set(other._names).intersection(self._names)
        if common_keys:
            raise KeyError('The key(s) `{}` already exist in the '
                           'DataContainer.'.format(', '.join(common_keys)))

        dicts = DataContainer.to_datasets(self)
        dicts.extend(DataContainer.to_datasets(other))
        return DataContainer(dicts)

    def __iter__(self):
        """
        Iterate through configuration object keys.

        Yields
        ------
        key
            A key in the container dictionary
        """
        for key in self.keys():
            yield key

    @staticmethod
    def to_datasets(data_container):
        """
        Convert a DataContainer object to a list of dataset dictionaries
        with keys {`name`, `path`, `frame`}.

        Parameters
        ----------
        data_container : DataContainer
            A DataContainer object.

        Returns
        -------
        datasets_dict : list of dicts
            A list of dataset dictionaries.
        """
        dataset_dicts = []
        for name in data_container.keys():
            dataset_dict = {'name': name,
                            'path': data_container.get_path(name),
                            'frame': data_container.get_frame(name)}
            dataset_dicts.append(dataset_dict)
        return dataset_dicts

    def add_dataset(self, dataset_dict, update=False):
        """
        Update or add a new DataFrame to the instance.

        Parameters
        ----------
        dataset_dict : pd.DataFrame
            The dataset dictionary to add.
        update : bool, optional
            Update an existing DataFrame, if True.
            Defaults to False.
        """
        name = dataset_dict['name']
        data_frame = dataset_dict['frame']
        path = dataset_dict.get('path')

        if not update:
            if name in self._names:
                raise KeyError('The name {} already exists in the '
                               'DataContainer dictionary.'.format(name))

        if name not in self._names:
            self._names.append(name)

        self._dataframes[name] = data_frame
        self._data_paths[name] = path

        self.__setattr__(name, data_frame)

    def get_path(self, key, default=None):
        """
        Get path, given key.

        Parameters
        ----------
        key : str
            Name for the data.

        Returns
        -------
        path : str
            Path to the data.
        """
        if key not in self._names:
            return default
        return self._data_paths[key]

    def get_frame(self, key, default=None):
        """
        Get frame, given key.

        Parameters
        ----------
        key : str
            Name for the data.
        default
            The default argument, if the frame does not exist

        Returns
        -------
        frame : pd.DataFrame
            The DataFrame.
        """
        if key not in self._names:
            return default
        return self._dataframes[key]

    def get_frames(self, prefix=None, suffix=None):
        """
        Get all data frames in the container that have
        a specified prefix and/or suffix. Note that
        the selection by prefix or suffix will be
        case-insensitive.

        Parameters
        ----------
        prefix : str or None, optional
            Only return frames with the given prefix.
            If None, then do not exclude any frames based
            on their prefix.
            Defaults to None.
        suffix : str or None, optional
            Only return frames with the given suffix.
            If None, then do not exclude any frames based
            on their suffix.
            Defaults to None.

        Returns
        -------
        frames : dict
            A dictionary with all of the data frames
            that contain the specified prefix and suffix.
            The keys are the names of the data frames.
        """
        if prefix is None:
            prefix = ''

        if suffix is None:
            suffix = ''

        names = [name for name in self._names if
                 name.lower().startswith(prefix) and
                 name.lower().endswith(suffix)]

        frames = {}
        for name in names:
            frames[name] = self._dataframes[name]
        return frames

    def keys(self):
        """
        Return keys as a list.

        Returns
        -------
        keys : list
            A list of keys in the Configuration object.
        """
        return self._names

    def values(self):
        """
        Return values as a list.

        Returns
        -------
        values : list
            A list of values in the Configuration object.
        """
        return [self._dataframes[name] for name in self._names]

    def items(self):
        """
        Return items as a list of tuples.

        Returns
        -------
        items : list of tuples
            A list of (key, value) tuples in the Configuration object.
        """
        return [(name, self._dataframe[name]) for name in self._names]

    def drop(self, name):
        """
        Drop a given data frame from the
        container.

        Parameters
        ----------
        name : str
            The name of the data frame to drop from the
            container object.

        Returns
        -------
        self
        """
        if name not in self:
            warnings.warn('The name `{}` is not in the '
                          'container. No data frames will '
                          'be dropped.'.format(name))
        else:
            self._names.remove(name)
            self._dataframes.pop(name)
            self._data_paths.pop(name)
        return self

    def rename(self, name, new_name):
        """
        Rename a given data frame in the
        container.

        Parameters
        ----------
        name : str
            The name of the current data frame
            in the container object.
        new_name : str
            The the new name for the data frame
            in the container object.

        Returns
        -------
        self
        """
        if name not in self:
            warnings.warn('The name `{}` is not in the '
                          'container and cannot '
                          'be renamed.'.format(name))
        else:
            frame = self._dataframes[name]
            path = self._data_paths[name]
            self.add_dataset({'name': new_name,
                              'frame': frame,
                              'path': path},
                             update=True)
            self.drop(name)
        return self

    def copy(self, deep=True):
        """
        Create a copy of the DataContainer object.

        Parameters
        ----------
        deep : bool, optional
            If True, create a deep copy of the
            underlying data frames.
            Defaults to True.
        """
        if deep:
            return deepcopy(self)
        return copy(self)
