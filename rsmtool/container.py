"""
Classes for storing any kind of data contained
in a pd.DataFrame object.

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:date: 10/25/2017
:organization: ETS
"""


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
            A list of dataset dicts. Each dict should be in the following format:
            {'name': 'name_of_dataset', 'frame' <pd.DataFrame object>}
        """
        self._names = []
        self._dataframes = {}
        self._data_paths = {}

        if datasets is not None:
            for dataset_dict in datasets:
                self.add_dataset(dataset_dict,
                                 update=False)

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
            raise KeyError('The key(s) `{}` already exist'
                           'in the DataContainer.'.format(', '.join(common_keys)))

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

    def copy(self, deep=True):
        """
        Create a copy of the DataContainer object

        Parameters
        ----------
        deep : bool, optional
            If True, create a deep copy.
            Defaults to True.
        """
        dataset_list = []
        for name in self.keys():
            frame = self[name].copy(deep=deep)
            dataset_dict = {'name': name,
                            'frame': frame}
            dataset_list.append(dataset_dict)

        return DataContainer(dataset_list)
