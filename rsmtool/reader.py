"""
Classes for reading data files (or dictionaries) into DataContainer objects.

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

import warnings
from functools import partial
from os.path import abspath, exists, join, splitext

import pandas as pd

from .container import DataContainer

# allow older versions of pandas to work
try:
    from pandas.io.common import DtypeWarning
except ImportError:
    from pandas.errors import DtypeWarning


def read_jsonlines(filename, converters=None):
    """
    Read a data file in .jsonlines format into a data frame.

    Normalize nested jsons with up to one level of nesting.

    Parameters
    ----------
    filename: str
        Name of file to read.
    converters : dict, optional
        A dictionary specifying how the types of the columns
        in the file should be converted. Specified in the same
        format as for ``pandas.read_csv()``.
        Defaults to ``None``.

    Returns
    -------
    df : pandas DataFrame
         Data frame containing the data in the given file.
    """
    try:
        df = pd.read_json(filename, orient="records", lines=True, dtype=converters)
    except ValueError:
        raise ValueError(
            "The jsonlines file is not formatted correctly. "
            "Please check that each line ends with a comma, "
            "there is no comma at the end of the last line, "
            "and that all quotes match."
        )

    # make sure we didn't get a plain json
    if type(df.columns) == pd.RangeIndex:
        raise ValueError(
            f"It looks like {filename} is a simple json file. Please "
            f"check documentation (for the expected file format"
        )

    dfs = []
    for column in df:
        # let's try to normalize this column
        try:
            df_column = pd.json_normalize(df[column])

            # Starting with Pandas v1.3, we get an empty data frame
            # if the column does not contain a nested json.
            # If this is the case, we simply copy the column.
            if df_column.empty:
                df_column = df[column].copy()

        # Pandas <v1.3 will raise an attribute error instead,
        # so we'll catch that too
        except AttributeError:
            df_column = df[column].copy()

        dfs.append(df_column)

    df = pd.concat(dfs, axis=1)

    return df


def try_to_load_file(filename, converters=None, raise_error=False, raise_warning=False, **kwargs):
    """
    Read a single file, if it exists.

    Optionally raises an error or warning if the file cannot be found.
    Otherwise, returns ``None``.

    Parameters
    ----------
    filename : str
        Name of file to read.
    converters : dict, optional
        A dictionary specifying how the types of the columns
        in the file should be converted. Specified in the same
        format as for ``pandas.read_csv()``.
        Defaults to ``None``.
    raise_error : bool, optional
        Raise an error if the file cannot be located.
        Defaults to ``False``.
    raise_warning : bool, optional
        Raise a warning if the file cannot be located.
        Defaults to ``False``.

    Returns
    -------
    df : pandas DataFrame or ``None``
        DataFrame containing the data in the given file,
        or ``None`` if the file does not exist.

    Raises
    ------
    FileNotFoundError
        If ``raise_error`` is ``True`` and the file cannot be located.
    """
    if exists(filename):
        return DataReader.read_from_file(filename, converters, **kwargs)

    message = f"The file '{filename}' could not be located."
    if raise_error:
        raise FileNotFoundError(message)

    if raise_warning:
        warnings.warn(message)


class DataReader:
    """Class to generate DataContainer objects."""

    def __init__(self, filepaths, framenames, file_converters=None):
        """
        Initialize a DataReader object.

        Parameters
        ----------
        filepaths : list of str
            A list of paths to files that are to be read in.
        framenames : list of str
            A list of names for the data sets to be included in the container.
        file_converters : dict of dicts, optional
            A dictionary of file converter dicts. The keys are the data set
            names and the values are the converter dictionaries to be applied
            to the corresponding data set.
            Defaults to ``None``.

        Raises
        ------
        AssertionError
            If ``len(filepaths)`` does not equal ``len(framenames)``.
        ValueError
            If ``file_converters`` is not a dictionary or if any of its
            values is not a dictionary.
        NameError
            If a key in ``file_converters`` does not exist in ``framenames``.
        ValueError
            If any of the specified file paths is ``None``.
        """
        # Default datasets list
        self.datasets = []

        # Make sure filepaths length matches frame names length
        assert len(filepaths) == len(framenames)

        # Make sure that there are no Nones in the filepaths
        if None in filepaths:
            frames_with_no_path = [
                framenames[i] for i in range(len(framenames)) if filepaths[i] is None
            ]

            raise ValueError(f"No path specified for {' ,'.join(frames_with_no_path)}")

        # Assign names and paths lists
        self.dataset_names = framenames
        self.dataset_paths = filepaths

        # If `file_converters` exists, then
        # check to make sure it is the correct length
        # and add all elements to `file_converters` list
        if file_converters is not None:
            if not isinstance(file_converters, dict):
                raise ValueError(
                    f"The 'file_converters' argument must be a `dict`, "
                    f"not `{type(file_converters)}`."
                )

            for file_converter_name in file_converters:
                # Make sure file_converter name is in `dataset_names`
                if file_converter_name not in self.dataset_names:
                    raise NameError(
                        f"The file converter name ``{file_converter_name}`` "
                        f"does not exist in the dataset names that you passed."
                    )

                # Make sure file converter is a `dict`
                file_converter = file_converters[file_converter_name]
                if not isinstance(file_converter, dict):
                    raise ValueError(
                        f"Value for {file_converter_name} must be``dict`` "
                        f"not {type(file_converter)}"
                    )

        # Default file_converters dict
        self.file_converters = {} if file_converters is None else file_converters

    @staticmethod
    def read_from_file(filename, converters=None, **kwargs):
        """
        Read a CSV/TSV/XLSX/JSONLINES/SAS7BDAT file and return a data frame.

        Parameters
        ----------
        filename : str
            Name of file to read.
        converters : dict, optional
            A dictionary specifying how the types of the columns
            in the file should be converted. Specified in the same
            format as for `pandas.read_csv()``.
            Defaults to ``None``.

        Returns
        -------
        df : pandas DataFrame
            Data frame containing the data in the given file.

        Raises
        ------
        ValueError
            If the file has an unsuppored extension.
        pandas.errors.ParserError
            If the file is badly formatted or corrupt.

        Note
        ----
        Any additional keyword arguments are passed to the underlying
        pandas IO reader function.
        """
        file_extension = splitext(filename)[1].lower()

        if file_extension in [".csv", ".tsv"]:
            sep = "\t" if file_extension == ".tsv" else ","
            do_read = partial(pd.read_csv, sep=sep, converters=converters)
        elif file_extension == ".xlsx":
            do_read = partial(pd.read_excel, converters=converters)
        elif file_extension == ".sas7bdat":
            if "encoding" not in kwargs:
                encoding = "latin-1"
            else:
                encoding = kwargs.pop("encoding")
            do_read = partial(pd.read_sas, encoding=encoding)
        elif file_extension in [".jsonlines"]:
            do_read = partial(read_jsonlines, converters=converters)
        else:
            raise ValueError(
                f"RSMTool only supports files in .csv, .tsv, .xlsx, "
                f"and .sas7bdat formats. Input files should have one "
                f"of these extensions. The file you passed is: {filename}."
            )

        # ignore warnings about mixed data types for large files
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DtypeWarning)
            try:
                df = do_read(filename, **kwargs)
            except pd.errors.ParserError:
                raise pd.errors.ParserError(
                    f"Cannot read {filename}. Please check "
                    f"that it is not corrupt or in an incompatible "
                    f"format. (Try running dos2unix?)"
                )
        return df

    @staticmethod
    def locate_files(filepaths, configdir):
        """
        Locate an experiment file, or a list of experiment files.

        If the given path doesn't exist, then maybe the path is relative
        to the path of the config file. If neither exists, then return
        ``None``.

        Parameters
        ----------
        filepaths : str or list
            Name(s) of the experiment file we want to locate.
        configdir : str
            Path to the reference configuration directory
            (usually the directory of the config file)

        Returns
        --------
        retval :  str or list
            Absolute path to the experiment file or ``None``
            if the file could not be located. If ``filepaths``
            was a string, this method will return a string.
            Otherwise, it will return a list.

        Raises
        ------
        ValueError
            If ``filepaths`` is not a string or a list.
        """
        # the feature config file can be in the 'feature' directory
        # at the same level as the main config file
        if not (isinstance(filepaths, str) or isinstance(filepaths, list)):
            raise ValueError(
                f"The 'filepaths' argument must be a string or a list, " f"not {type(filepaths)}."
            )

        if isinstance(filepaths, str):
            filepaths = [filepaths]
            return_string = True
        else:
            return_string = False

        located_paths = []
        for filepath in filepaths:

            retval = None
            alternate_path = abspath(join(configdir, filepath))

            # if the given path exists as is, convert
            # that to an absolute path and return
            if exists(filepath):
                retval = abspath(filepath)

            # otherwise check if it exists relative
            # to the reference directory
            elif exists(alternate_path):
                retval = alternate_path

            located_paths.append(retval)

        if return_string:
            return located_paths[0]

        return located_paths

    def read(self, kwargs_dict=None):
        """
        Read all files contained in ``self.dataset_paths``.

        Parameters
        ----------
        kwargs_dict : dict of dicts, optional
            Any additional keyword arguments to pass to a particular DataFrame.
            These arguments will be passed to the pandas IO reader function.
            Defaults to ``None``.

        Returns
        -------
        datacontainer : container.DataContainer
            A data container object.

        Raises
        ------
        FileNotFoundError
            If any of the files in ``self.dataset_paths`` does not exist.
        """
        for idx, set_path in enumerate(self.dataset_paths):

            name = self.dataset_names[idx]
            converter = self.file_converters.get(name, None)

            if not exists(set_path):
                raise FileNotFoundError(f"The file {set_path} does not exist")

            if kwargs_dict is not None:
                kwargs = kwargs_dict.get(name, {})
            else:
                kwargs = {}

            dataframe = self.read_from_file(set_path, converter, **kwargs)

            # Add to list of datasets
            self.datasets.append({"name": name.strip(), "path": set_path, "frame": dataframe})

        return DataContainer(self.datasets)
