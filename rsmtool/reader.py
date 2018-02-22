"""
Classes for reading data files (or dictionaries)
and converting them to DataContainer objects.

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:date: 10/25/2017
:organization: ETS
"""

import warnings

from functools import partial
from os.path import (abspath,
                     exists,
                     join,
                     splitext)

import pandas as pd

from rsmtool.container import DataContainer


class DataReader:
    """
    A DataReader class to generate
    DataContainer objects
    """

    def __init__(self,
                 filepaths,
                 framenames,
                 file_converters=None):
        """
        Initialize DataReader object.

        Parameters
        ----------
        filepaths : list of str
            A list of paths to files that will be read into pd.DataFrames.
        filenames : list of str
            A list of names for the pd.DataFrames.
        file_converters : dict of dicts, optional
            A dictionary of file converter dicts.
            Defaults to None

        Raises
        ------
        AssertionError
            If length of filepaths is not equal to length of framenames.
        ValueError
            If any elements in file_converters are not dict.
        NameError
            If file converter name does not exist in the dataset.
        ValueError
            If filepath for a given file is None
        """

        # Default datasets list
        self.datasets = []

        # Make sure filepaths length matches frame names length
        assert len(filepaths) == len(framenames)

        # Make sure that there are no Nones in the filepaths
        if None in filepaths:
            frames_with_no_path = [framenames[i] for i in range(len(framenames))
                                   if filepaths[i] is None]

            raise ValueError("No path specified for "
                             "{}".format(' ,'.join(frames_with_no_path)))

        # Assign names and paths lists
        self.dataset_names = framenames
        self.dataset_paths = filepaths

        # If file_converters exists, then
        # check to make sure it is the correct length
        # and add all elements to file_converters list
        if file_converters is not None:

            # assert len(filepaths) == len(file_converters)

            if not isinstance(file_converters, dict):
                raise ValueError('The `file_converters` argument must be type ``dict``, '
                                 'not ``{}``.'.format(type(file_converters)))

            for file_converter_name in file_converters:

                # Make sure file_converter name is in `dataset_names`
                if file_converter_name not in self.dataset_names:
                    raise NameError('The file converter name ``{}`` '
                                    'does not exist in the '
                                    'dataset names that you '
                                    'passed.'.format(file_converter_name))

                # Make sure file converter is a `dict`
                file_converter = file_converters[file_converter_name]
                if not isinstance(file_converter, dict):
                    raise ValueError('Value for {} must be``dict`` ',
                                     'not {}'.format(file_converter_name,
                                                     type(file_converter)))

        # Default file_converters dict
        self.file_converters = {} if file_converters is None else file_converters

    @staticmethod
    def read_from_file(filename, converters=None, **kwargs):
        """
        Read a CSV/TSV/XLS/XLSX file and return a data frame.

        Parameters
        ----------
        filename : str
            Name of file to read.
        converters : None, optional
            A dictionary specifying how the types of the columns
            in the file should be converted. Specified in the same
            format as for ``pandas.read_csv()``.

        Returns
        -------
        df : pandas DataFrame
            Data frame containing the data in the given file.

        Raises
        ------
        ValueError
            If the file has an extension that we do not support
        pd.parser.CParserError
            If the file is badly formatted or corrupt.

        Note
        ----
        Keyword arguments are passed to the given `pandas`
        IO reader function.
        """

        file_extension = splitext(filename)[1].lower()

        if file_extension in ['.csv', '.tsv']:
            sep = '\t' if file_extension == '.tsv' else ','
            do_read = partial(pd.read_csv, sep=sep, converters=converters)
        elif file_extension in ['.xls', '.xlsx']:
            do_read = partial(pd.read_excel, converters=converters)
        else:
            raise ValueError("RSMTool only supports files in .csv, "
                             ".tsv or .xls/.xlsx format. "
                             "The file should have the extension "
                             "which matches its format. The file you "
                             "passed is: {}.".format(filename))

        # ignore warnings about mixed data types for large files
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=pd.io.common.DtypeWarning)
            try:
                df = do_read(filename, **kwargs)
            except pd.parser.CParserError:
                raise pd.parser.CParserError('Cannot read {}. Please check that it is '
                                             'not corrupt or in an incompatible format. '
                                             '(Try running dos2unix?)'.format(filename))
        return df

    @staticmethod
    def locate_files(filepaths, config_dir):
        """
        Try to locate an experiment file, or a list of experiment files.
        If the given path doesn't exist, then maybe the path is relative
        to the path of the config file. If neither exists, then return None.

        Parameters
        ----------
        filepath_or_paths : str or list
            Name of the experiment file we want to locate.
        config_dir : str
            Path to the experiment configuration file.

        Returns
        --------
        retval :  str or list
            Absolute path to the experiment file or None
            if the file could not be located. If the `filepaths` argument
            was a string, this method will return a string. Otherwise, it will
            return a list.

        Raises
        ------
        ValueError
            If filepaths  is not a string or list.
        """

        # the feature config file can be in the 'feature' directory
        # at the same level as the main config file

        if not (isinstance(filepaths, str) or
                isinstance(filepaths, list)):

            raise ValueError('The `filepaths` argument must be a '
                             'string or list, not {}.'.format(type(filepaths)))

        if isinstance(filepaths, str):
            filepaths = [filepaths]
            return_string = True
        else:
            return_string = False

        located_paths = []
        for filepath in filepaths:

            retval = None
            alternate_path = abspath(join(config_dir, filepath))

            # if the given path exists as is, convert
            # that to an absolute path and return
            if exists(filepath):
                retval = abspath(filepath)

            # otherwise check if it exists relative
            # to the directory that contains the main config file
            elif exists(alternate_path):
                retval = alternate_path

            located_paths.append(retval)

        if return_string:
            return located_paths[0]

        return located_paths

    def read(self,
             kwargs_dict=None):
        """
        Read all files passed to the constructor.


        Parameters
        ----------
        kwargs_dict : dict of dicts, optional
            Any additional keyword arguments to pass to a particular DataFrame.
            These arguments will be passed to the `pandas` IO reader function.
            Defaults to None.

        Returns
        -------
        datacontainer : DataContainer
            A DataContainer object.
        """

        for idx, set_path in enumerate(self.dataset_paths):

            name = self.dataset_names[idx]
            converter = self.file_converters.get(name, None)

            if not exists(set_path):
                raise FileNotFoundError('The file {} does not exist'.format(set_path))

            if kwargs_dict is not None:
                kwargs = kwargs_dict.get(name, {})
            else:
                kwargs = {}

            dataframe = self.read_from_file(set_path, converter, **kwargs)

            # Add to list of datasets
            self.datasets.append({'name': name.strip(),
                                  'path': set_path,
                                  'frame': dataframe})

        return DataContainer(self.datasets)
