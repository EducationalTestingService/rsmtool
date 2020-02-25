"""
Classes related to parsing configuration files
and creating configuration objects.

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

import functools
import json
import logging
import re
import warnings

from copy import copy, deepcopy
from collections import Counter
from configparser import ConfigParser

from os import getcwd, makedirs
from os.path import (abspath,
                     basename,
                     dirname,
                     join,
                     splitext)

from ruamel import yaml

from rsmtool import HAS_RSMEXTRA
from rsmtool.utils import parse_json_with_comments
from rsmtool.utils import (DEFAULTS,
                           CHECK_FIELDS,
                           LIST_FIELDS,
                           BOOLEAN_FIELDS,
                           MODEL_NAME_MAPPING,
                           FIELD_NAME_MAPPING,
                           ID_FIELDS,
                           is_skll_model)

from skll import Learner
from skll.metrics import SCORERS

if HAS_RSMEXTRA:
    from rsmextra.settings import (default_feature_subset_file,
                                   default_feature_sign)


def deprecated_positional_argument():
    """
    This decorator allows the Configuration class to:

    (a) accept the old method of specifying the now-deprecated ``filepath`` positional argument,
    (b) accept the new method of specifying ``configdir`` and ``filename`` keyword arguments, but
    (c) disallow using the old and the new methods in the same call

    Adapted from: https://stackoverflow.com/a/49802489
    """
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # if we received two positional arguments
            if len(args) > 2:
                # if we also received a keyword argument for filepath
                # or configdir, raise an error
                if 'filename' in kwargs:
                    raise ValueError("Cannot specify both the deprecated filepath "
                                     "positional argument and the new-style filename "
                                     "keyword argument.")
                if 'configdir' in kwargs:
                    raise ValueError("Cannot specify both the deprecated filepath "
                                     "positional argument and the new-style configdir "
                                     "keyword argument.")
                # raise deprecation warning
                warnings.warn("The filepath positional argument is deprecated and "
                              " will be removed in v8.0. Use the ``configdir`` and "
                              "``filename`` keyword arguments instead.",
                              DeprecationWarning)

                # split filepath into
                # configdir and filename
                filepath = args[-1]
                kwargs['filename'] = basename(filepath)
                kwargs['configdir'] = dirname(abspath(filepath))
                # remove filepath from positional arguments
                args = args[:-1]
            return f(*args, **kwargs)
        return wrapper
    return decorator


class Configuration:
    """
    Configuration class, which encapsulates all of the
    configuration parameters and methods to access these
    parameters.
    """

    @deprecated_positional_argument()
    def __init__(self,
                 config_dict,
                 *,
                 configdir=None,
                 filename=None,
                 context='rsmtool'):
        """
        Create an object of the `Configuration` class.

        Note that usually the Configuration
        object used for RSMTool experiments is created using
        `ConfigurationParser.load_normalize_and_validate_config_from_dict()` or
        `ConfigurationParser.read_normalize_validate_and_process_config()`.
        You should directly instantiate a Configuration object only if
        you already have a normalized configuration dictionary
        (e.g., from previous RSMTool experiments).


        Parameters
        ----------
        config_dict : dict
            A dictionary of configuration parameters.
            The dictionary must be a valid configuration dictionary
            with default values filled as necessary.
        configdir : str, optional, keyword-only
            The reference path used to
            resolve any relative paths in the configuration
            object. When None, will be set during
            initialization to the current working directory.
            Defaults to None
        filename : str, optional, keyword-only
            The name of the configuration file.
            The file must be stored in configdir.
            This argument is not used in RSMTool and only added for
            backwards compatibility for the deprecated `filepath` attribute.
            Defaults to None.
        context : {'rsmtool', 'rsmeval', 'rsmcompare',
                   'rsmpredict', 'rsmsummarize'}
            The context of the tool.
            Defaults to 'rsmtool'.
        """

        self._config = config_dict
        self._context = context

        # set configdir to `cwd` if not given and let the user know
        if configdir is None:
            configdir = getcwd()
            logging.info("Configuration directory will be set to {}".format(configdir))
        else:
            configdir = abspath(configdir)

        self._configdir = configdir
        self._filename = filename

    def __contains__(self, key):
        """
        Check if configuration object
        contains a given key.

        Parameters
        ----------
        key : str
            Key to check in the Configuration object.

        Returns
        -------
        key_check : bool
            True if key in Configuration object, else False
        """
        return key in self._config

    def __getitem__(self, key):
        """
        Get value, given key.

        Parameters
        ----------
        key : str
            Key to check in the Configuration object

        Returns
        -------
        value
            The value in the Configuration object dictionary.
        """
        return self._config[key]

    def __setitem__(self, key, value):
        """
        Set value, given key.

        Parameters
        ----------
        key : str
            Key to check in the Configuration object.
        value
            A value to be set on the key.
        """
        self._config[key] = value

    def __len__(self):
        """
        Return the length of the Configuration dictionary.

        Returns
        -------
        length : int
            The length of the config dictionary (i.e. number of elements)
        """
        return len(self._config)

    def __str__(self):
        """
        Return string representation of the object keys
        as comma-separated list.

        Returns
        -------
        config_names : str
            A comma-separated list of names from the config dictionary.
        """
        return ', '.join(self._config)

    def __iter__(self):
        """
        Iterate through configuration object keys.

        Yields
        ------
        key
            A key in the config dictionary
        """
        for key in self.keys():
            yield key

    @property
    def filepath(self):
        """
        Get file path to the configuration file.

        .. deprecated:: 8.0

            ``filepath`` will be removed in RSMTool v8.0. Use ``configdir`` and
            ``filename`` instead.

        Returns
        -------
        filepath : str
            The path for the config file.
        """
        warnings.warn("The `filepath` attribute of the Configuration "
                      "object will be removed in RSMTool v8.0."
                      "Use the `configdir` and `filename` attributes if you "
                      "need the full path to the "
                      "configuration file", DeprecationWarning)
        filepath = join(self.configdir, self.filename)
        return filepath

    @filepath.setter
    def filepath(self, new_path):
        """
        Set a new file path to configuration file.

        .. deprecated:: 8.0
        ``filepath`` will be removed in RSMTool v8.0. Use ``configdir`` and
        ``filename`` instead.

        Parameters
        ----------
        new_path : str
            A new file path for the Configuration object.
        """
        warnings.warn("The `filepath` attribute of the Configuration "
                      "object will be removed in RSMTool 8.0 "
                      "use `configdir` and `filename` if you "
                      "need to set a new path to the "
                      "configuration file", DeprecationWarning)
        new_filename = basename(new_path)
        new_configdir = dirname(abspath(new_path))
        self._filename = new_filename
        self._configdir = new_configdir

    @property
    def configdir(self):
        """
        Get the path to the configuration reference directory that
        will be used to resolve any relative paths in
        the configuration.

        Returns
        -------
        configdir : str
            The path to the configuration reference directory
        """
        return self._configdir

    @configdir.setter
    def configdir(self, new_path):
        """
        Set a new configuration reference directory

        Parameters
        ----------
        new_path : str
            Path to the new configuration reference
            directory used to resolve any relative paths
            in the configuration object.
        """

        if new_path is None:
            raise ValueError("The `configdir` attribute cannot be set to `None` ")
        self._configdir = abspath(new_path)

    @property
    def filename(self):
        """
        Get the name of the configuration file.

        Returns
        -------
        filename : str
            The name of the configuration file
        """
        return self._filename

    @filename.setter
    def filename(self, new_path):
        """
        Set a new name of the configuration file

        Parameters
        ----------
        new_name : str
            New name of the configuration file
        """
        self._filename = new_path

    @property
    def context(self):
        """
        Get the context.
        """
        return self._context

    @context.setter
    def context(self, new_context):
        """
        Set a new context

        Parameters
        ----------
        new_context : str
            A new context  for the
            configuration object.
        """
        self._context = new_context

    def get(self, key, default=None):
        """
        Get value or default, given key.

        Parameters
        ----------
        key : str
            Key to check in the Configuration object.
        default, optional
            The default value to return, if no key exists.
            Defaults to None.

        Returns
        -------
        value
            The value in the Configuration object dictionary.
        """
        return self._config.get(key, default)

    def to_dict(self):
        """
        Get a dictionary representation of the Configuration object.

        Returns
        -------
        config : dict
            The configuration dictionary.
        """
        return self._config

    def keys(self):
        """
        Return keys as a list.

        Returns
        -------
        keys : list of str
            A list of keys in the Configuration object.
        """
        return [k for k in self._config.keys()]

    def values(self):
        """
        Return values as a list.

        Returns
        -------
        values : list
            A list of values in the Configuration object.
        """
        return [v for v in self._config.values()]

    def items(self):
        """
        Return items as a list of tuples.

        Returns
        -------
        items : list of tuples
            A list of (key, value) tuples in the
            configuration object.
        """
        return [(k, v) for k, v in self._config.items()]

    def pop(self, key, default=None):
        """
        Remove and returns an element from
        the object having the given key.

        Parameters
        ----------
        key : str
            Key to pop in the configuration object.
        default, optional
            The default value to return, if no key exists.
            Defaults to None.

        Returns
        -------
        value
            The value removed from the object.
        """
        return self._config.pop(key, default)

    def copy(self, deep=True):
        """
        Return a copy of the object.

        Parameters
        ----------
        deep : bool, optional
            Whether to perform a deep copy.
            Defaults to True.

        Returns
        -------
        copy : Configuration
            A new configuration object.
        """
        if deep:
            return deepcopy(self)
        return copy(self)

    def save(self, output_dir=None):
        """
        Save the configuration file to the output directory specified.

        Parameters
        ----------
        output_dir : str
            The path to the output directory.
        """

        # save a copy of the main config into the output directory
        if output_dir is None:
            output_dir = getcwd()

        # Create output directory, if it does not exist
        output_dir = join(output_dir, 'output')
        makedirs(output_dir, exist_ok=True)

        id_field = ID_FIELDS[self._context]
        outjson = join(output_dir,
                       '{}_{}.json'.format(self._config[id_field],
                                           self._context))

        expected_fields = (CHECK_FIELDS[self._context]['required'] +
                           CHECK_FIELDS[self._context]['optional'])

        output_config = {k: v for k, v in self._config.items() if k in expected_fields}
        with open(outjson, 'w') as outfile:
            json.dump(output_config, outfile, indent=4, separators=(',', ': '))

    def check_exclude_listwise(self):
        """
        Check if we are excluding candidates
        based on number of responses, and
        add this to the configuration file

        Returns
        -------
        exclude_listwise : bool
            Whether to exclude list-wise
        """
        exclude_listwise = False
        if self._config.get('min_items_per_candidate'):
            exclude_listwise = True
        return exclude_listwise

    def check_flag_column(self,
                          flag_column='flag_column',
                          partition='unknown'):
        """
        Make sure the `flag_column` field is in the correct format. Get
        flag columns and values for filtering if any and convert single
        values to lists. Raises an exception if `flag_column` is not
        correctly specified.

        Parameters
        ----------
        flag_column : str
            The flag column to check. Currently used fields are `flag_column` or
            `flag_column_test`.
            Defaults to 'flag_column'.

        partition: {'train', 'test', 'both', 'unknown'}
            Partition which is filtered based on the flag column.
            This is used to display more helpful warning messages.
            Defaults to 'both'

        Returns
        -------
        new_filtering_dict : dict
            Properly formatted `flag_column` dictionary.

        Raises
        ------
        ValueError
            If the `flag_column` is not  a dictionary

            If `partition` value if not in the expected list

            If `partition` value does not match the `flag_column`
        """
        config = self._config

        new_filter_dict = {}

        flag_message = {'train': 'training',
                        'test': 'evaluating',
                        'both': 'training and evaluating',
                        'unknown': 'training and/or evaluating'}

        if partition not in flag_message:
            raise ValueError("Unknown value for partition: {} "
                             "This must be one of the following: {}."
                             "".format(partition, ','.join(flag_message.keys())))

        if flag_column == 'flag_column_test':
            if partition in ['both', 'train']:
                raise ValueError("The conditions specified in `flag_column_test` "
                                 "can only be applied to the evaluation partition.")

        if config.get(flag_column):

            original_filter_dict = config[flag_column]

            # first check that the value is a dictionary
            if not isinstance(original_filter_dict, dict):
                raise ValueError("`flag_column` must be a dictionary. "
                                 "Please refer to the documentation for "
                                 "further information.")

            for column in original_filter_dict:

                # if we were given a single value, convert it to list
                if not isinstance(original_filter_dict[column], list):
                    new_filter_dict[column] = [original_filter_dict[column]]
                    logging.warning("The filtering condition {}"
                                    " for column {} was converted "
                                    "to list. Only responses where "
                                    "{} == {} will be used for "
                                    "{} the "
                                    "model. You can ignore this "
                                    "warning if this is the correct "
                                    "interpretation of your "
                                    "configuration settings"
                                    ".".format(original_filter_dict[column],
                                               column,
                                               column,
                                               original_filter_dict[column],
                                               flag_message[partition])
                                    )
                else:
                    new_filter_dict[column] = original_filter_dict[column]

                    model_eval = ', '.join(map(str,
                                               original_filter_dict[column]))
                    logging.info("Only responses where "
                                 "{} equals one of the following values "
                                 "will be used for {} the model: "
                                 "{}.".format(column,
                                              flag_message[partition],
                                              model_eval))
        return new_filter_dict

    def get_trim_min_max(self):
        """
        This function is kept for backwards compatibility.
        It is now replaced by get_trim_min_max_tolerance
        and will be deprecated in future versions.

        Returns
        -------
        spec_trim_min : float
            Specified trim min value
        spec_trim_max : float
            Specified trim max value
        """
        logging.warning("get_trim_min_max method has been replaced by  "
                        "get_trim_min_max_tolerance() and will "
                        "not be supported in future releases.")
        (trim_min, trim_max, tolerance) = self.get_trim_min_max_tolerance()
        return (trim_min, trim_max)

    def get_trim_min_max_tolerance(self):
        """
        Get the specified trim min and max,
        and trim_tolerance if any,
        and make sure they are numeric.

        Returns
        -------
        spec_trim_min : float
            Specified trim min value
        spec_trim_max : float
            Specified trim max value
        spec_trim_tolerance: float
            specified trim tolerance value
        """
        config = self._config

        spec_trim_min = config.get('trim_min', None)
        spec_trim_max = config.get('trim_max', None)
        spec_trim_tolerance = config.get('trim_tolerance', None)

        if spec_trim_min:
            spec_trim_min = float(spec_trim_min)
        if spec_trim_max:
            spec_trim_max = float(spec_trim_max)
        if spec_trim_tolerance:
            spec_trim_tolerance = float(spec_trim_tolerance)
        return (spec_trim_min, spec_trim_max, spec_trim_tolerance)

    def get_default_converter(self):
        """
        Get the default converter dictionary for reader.

        Returns
        -------
        default_converter : dict
            The default converter for a train or test file.
        """
        string_columns = [self._config['id_column']]

        candidate = self._config.get('candidate_column')
        if candidate is not None:
            string_columns.append(candidate)

        subgroups = self._config.get('subgroups')
        if subgroups:
            string_columns.extend(subgroups)

        return dict([(column, str) for column in string_columns if column])

    def get_names_and_paths(self, keys, names):
        """
        Get a a list of values, given keys.
        Remove any values that are None.

        Parameters
        -------
        keys : list
            A list of keys whose values to retrieve.
        names : list
            The default value to use if key cannot be found.
            Defaults to None.

        Returns
        -------
        values : list
            The list of values.

        Raises
        ------
        ValueError
            If there are any duplicate keys or names.
        """

        assert len(keys) == len(names)

        # Make sure keys are not duplicated
        if not len(set(keys)) == len(keys):
            raise ValueError('The ``keys`` must be unique. However, the '
                             'following duplicate keys were found: {}.'
                             ''.format(', '.join([key for key, val in Counter(keys).items()
                                                  if val > 1])))
        # Make sure names are not duplicated
        if not len(set(names)) == len(names):
            raise ValueError('The``names`` must be unique. However, the '
                             'following duplicate names were found: {}.'
                             ''.format(', '.join([name for name, val in Counter(names).items()
                                                  if val > 1])))
        existing_names = []
        existing_paths = []
        for idx, key in enumerate(keys):

            path = self._config.get(key)

            # if the `features` field is a list,
            # do not include it in the container
            if key == 'features':
                if isinstance(path, list):
                    continue

            if path is not None:
                existing_paths.append(path)
                existing_names.append(names[idx])

        return existing_names, existing_paths


class ConfigurationParser:
    """
    A `ConfigurationParser` class to create a
    `Configuration` object.
    """

    def __init__(self):

        # Set configuration object to None
        self._config = None
        self._filename = None
        self._configdir = None

    def _check_config_is_loaded(self):
        """
        Check to make sure a configuration file
        or dictionary was loaded; otherwise,
        raise ``NameError``.

        Raises
        ------
        NameError
            If no configuration file or dictionary was loaded.
        """
        if self._config is None:
            raise NameError('No configuration file was loaded '
                            'Make sure to load a configuration file '
                            'from a dict using the `load_config_from_dict()` '
                            'method or use the `read_config_from_file()` method '
                            'with the appropriate sub-class object to read from '
                            'a file. You can use the `get_configparser` class '
                            'method to instantiate the appropriate sub-class '
                            'object for reading either `.json` or `.cfg` files.')


    @classmethod
    def get_configparser(cls, filepath, *args, **kwargs):
        """
        Get the correct `ConfigurationParser` object,
        based on the file extension.

        Parameters
        ----------
        filepath : str
            The path to the configuration file.

        Returns
        -------
        config : ConfigurationParser
            The configuration parser object.

        Raises
        ------
        ValueError
            If config file is not .json or .cfg.
        """
        _, extension = splitext(filepath)
        if extension.lower() not in CONFIG_TYPE:
            raise ValueError('Configuration file must be '
                             'in either `.json` or `.cfg`'
                             'format. You specified: {}.'.format(extension))

        return CONFIG_TYPE[extension.lower()](*args, **kwargs)

    @staticmethod
    def check_id_fields(id_field_values):
        """
        Check whether the ID fields in the given dictionary
        are properly formatted, i.e., they ::
        - do not contain any spaces
        - are shorter than 200 characters

        Parameters
        ----------
        id_field_values : dict
            A dictionary containing the ID fields names
            as the keys and the value from the configuration
            file as the value.

        Raises
        ------
        ValueError
            If the values for the ID fields in the given
            dictionary are not formatted correctly.
        """

        for id_field, id_field_value in id_field_values.items():
            if len(id_field_value) > 200:
                raise ValueError("{} is too long (must be "
                                 "<=200 characters)".format(id_field))

            if re.search(r'\s', id_field_value):
                raise ValueError("{} cannot contain any "
                                 "spaces".format(id_field))

    def load_config_from_dict(self,
                              config_dict,
                              configdir=None):
        """
        Load configuration dictionary.

        Parameters
        ----------
        config_dict : dict
            A dictionary containing the configuration
            parameters to parse.
        configdir : str, optional
            Path to the reference directory used to resolve
            any relative path in the dictionary. If not specified,
            the current working directory will be used.
        filename: str, optional


        Raises
        ------
        TypeError
            If `config_dict` is not a ``dict``
        AttributeError
            If config has already been assigned.
        """
        if not isinstance(config_dict, dict):
            raise TypeError('The `config_dict` must be a dictionary.')

        if self._config is None:
            self._config = config_dict
        else:
            raise AttributeError('A configuration dictionary has already'
                                 'been assigned. You cannot assign another'
                                 'dictionary.')

        if configdir is None:
            configdir = getcwd()

        self._configdir = abspath(configdir)
        logging.info("Configuration directory will be set to {}".format(self._configdir))

        # set filename to none since there was no configuration file.
        # If the user for some reason wants
        # to set it, they can do so explicitly.
        self._filename = None


    def load_normalize_and_validate_config_from_dict(self,
                                                     config_dict,
                                                     configdir=getcwd(),
                                                     context='rsmtool'):
        """
        Load configuration dictionary.

        Parameters
        ----------
        config_dict : dict
            A dictionary containing the configuration
            parameters to parse.
        configdir : str, optional
            Path to the reference directory used to resolve
            any relative path in the dictionary.
            Defaults to the current working directory.
        context : str, optional
            Context of the tool in which we are validating.
            Possible values are ::

                {'rsmtool', 'rsmeval',
                 'rsmpredict', 'rsmcompare', 'rsmsummarize'}

            Defaults to 'rsmtool'.

        Returns
        -------
        config_obj : Configuration
            A configuration object
        """
        self.load_config_from_dict(config_dict, configdir)
        return self.normalize_validate_and_process_config(context=context)




    def read_config_from_file(self, filepath):
        """
        Read the configuration file.

        Parameters
        ----------
        filepath : str
            The path to the configuration file.

        Raises
        ------
        NotImplementedError
            This method must be implemented in subclass.
        """
        raise NotImplementedError("The method `read_config_from_file()` "
                                  "is only implemented in the subclasses "
                                  "``CFGConfigurationParser`` and "
                                  "``JSONConfigurationParser``. "
                                  "You can use the class method "
                                  "`get_configparser()` to retrieve "
                                  "the correct configuration parser object "
                                  "for parsing JSON or CFG files.")

    def normalize_config(self, inplace=True):
        """
        Normalize the field names in `self._config` or `config` in order to
        maintain backwards compatibility with old configuration files.

        Parameters
        ----------
        inplace : bool
            Maintain the state of the config object produced by
            this method.
            Defaults to True.

        Returns
        -------
        new_config : Configuration
            A normalized configuration object

        Raises
        ------
        ValueError
            If no JSON configuration object exists, or if value passed for
            `use_scaled_predictions` is in the wrong format.
        """

        # Check to make sure a configuration file
        # or dictionary has been loaded.
        self._check_config_is_loaded()

        # Get the parameter dictionary
        config = self._config

        # Create a new JSON object with the normalized field names
        new_config = {}

        for field_name in config:

            if field_name in FIELD_NAME_MAPPING:
                norm_field_name = FIELD_NAME_MAPPING[field_name]
                warnings.warn("""The field name "{}" is deprecated """
                              """and will be removed  in a future """
                              """release, please use the """
                              """new field name "{}" """
                              """instead.""".format(field_name,
                                                    norm_field_name),
                              category=DeprecationWarning)
            else:
                norm_field_name = field_name

            new_config[norm_field_name] = config[field_name]

        # Convert old values for prediction scaling:
        if 'use_scaled_predictions' in new_config:
            if new_config['use_scaled_predictions'] in ['scale', True]:
                new_config['use_scaled_predictions'] = True
            elif new_config['use_scaled_predictions'] in ['raw', False]:
                new_config['use_scaled_predictions'] = False
            else:
                raise ValueError("Please use the new format "
                                 "to specify prediction scaling:\n "
                                 "'use_scaled_predictions': true/false")

        # Convert old model names to new ones, if we have them
        if 'model' in new_config:
            model_name = new_config['model']

            if model_name == 'empWtDropNeg':

                # If someone is using `empWtDropNeg`, we tell them that it is
                # no longer available and they should be using NNLR instead.
                logging.error("""The model name "empWtDropNeg" is """
                              """no longer available, please use the """
                              """equivalent model "NNLR" instead.""")

            # Otherwise, just raise a deprecation warning if they are using
            # an old model name
            elif model_name in MODEL_NAME_MAPPING:
                norm_model_name = MODEL_NAME_MAPPING[model_name]
                warnings.warn("""The model name "{}" is deprecated """
                              """and will be removed  in a future """
                              """release, please use the new model """
                              """name "{}" instead.""".format(model_name,
                                                              norm_model_name),
                              category=DeprecationWarning)
                new_config['model'] = norm_model_name

        if inplace:
            self._config = new_config
        return Configuration(self._config,
                             configdir=self._configdir,
                             filename=self._filename,)

    def validate_config(self, context='rsmtool', inplace=True):
        """
        Ensure that all required fields are specified, add default values
        values for all unspecified fields, and ensure that all specified
        fields are valid.

        Parameters
        ----------
        context : str, optional
            Context of the tool in which we are validating.
            Possible values are ::

                {'rsmtool', 'rsmeval',
                 'rsmpredict', 'rsmcompare', 'rsmsummarize'}

            Defaults to 'rsmtool'.
        inplace : bool
            Maintain the state of the config object produced by
            this method.
            Defaults to True.

        Returns
        -------
        config_obj : Configuration
            A configuration object

        Raises
        ------
        ValueError
            If config does not exist, and no config passed.
        """

        # Check to make sure a configuration file
        # or dictionary has been loaded.
        self._check_config_is_loaded()

        # Get the parameter dictionary
        new_config = self._config

        # 1. Check to make sure all required fields are specified
        required_fields = CHECK_FIELDS[context]['required']

        for field in required_fields:
            if field not in new_config:
                raise ValueError("The config file must "
                                 "specify '{}'".format(field))

        # 2. Add default values for unspecified optional fields
        # for given RSMTool context
        defaults = DEFAULTS

        for field in defaults:
            if field not in new_config:
                new_config[field] = defaults[field]

        # 3. Check to make sure no unrecognized fields are specified
        for field in new_config:
            if field not in defaults and field not in required_fields:
                raise ValueError("Unrecognized field '{}'"
                                 " in json file".format(field))

        # 4. Check to make sure that the ID fields that will be
        # used as part of filenames formatted correctly
        id_field = ID_FIELDS[context]
        id_field_values = {id_field: new_config[id_field]}

        self.check_id_fields(id_field_values)

        # 5. Check that the feature file and feature subset/subset file are not
        # specified together
        msg = ("You cannot specify BOTH \"features\" and \"{}\". "
               "Please refer to the \"Selecting Feature Columns\" "
               "section in the documentation for more details.")
        if new_config['features'] and new_config['feature_subset_file']:
            msg = msg.format("feature_subset_file")
            raise ValueError(msg)
        if new_config['features'] and new_config['feature_subset']:
            msg = msg.format("feature_subset")
            raise ValueError(msg)

        # 6. Check for fields that require feature_subset_file and try
        # to use the default feature file
        if (new_config['feature_subset'] and
                not new_config['feature_subset_file']):

            # Check if we have the default subset file from rsmextra
            if HAS_RSMEXTRA:
                default_basename = basename(default_feature_subset_file)
                new_config['feature_subset_file'] = default_feature_subset_file
                logging.warning("You requested feature subsets but did not "
                                "specify any feature file. "
                                "The tool will use the default "
                                "feature file {} available via "
                                "rsmextra".format(default_basename))
            else:
                raise ValueError("If you want to use feature subsets, you "
                                 "must specify a feature subset file")

        if new_config['sign'] and not new_config['feature_subset_file']:

            # Check if we have the default subset file from rsmextra
            if HAS_RSMEXTRA:
                default_basename = basename(default_feature_subset_file)
                new_config['feature_subset_file'] = default_feature_subset_file
                logging.warning("You specified the expected sign of "
                                "correlation but did not specify a feature "
                                "subset file. The tool will use "
                                "the default feature subset file {} "
                                "available via "
                                "rsmextra".format(default_basename))
            else:
                raise ValueError("If you want to specify the expected sign of "
                                 " correlation for each feature, you must "
                                 "specify a feature subset file")

        # Use the default sign if we are using the default feature file
        # and sign has not been specified in the config file
        if HAS_RSMEXTRA:
            default_feature = default_feature_subset_file
            if (new_config['feature_subset_file'] == default_feature and
                    not new_config['sign']):
                new_config['sign'] = default_feature_sign

        # 7. Check for fields that must be specified together
        if (new_config['min_items_per_candidate'] and
                not new_config['candidate_column']):
            raise ValueError("If you want to filter out candidates with "
                             "responses to less than X items, you need "
                             "to specify the name of the column which "
                             "contains candidate IDs.")

        # 8. Check that if "skll_objective" is specified, it's
        # one of the metrics that SKLL allows for AND that it is
        # specified for a SKLL model and _not_ a built-in
        # linear regression model
        if new_config['skll_objective']:
            if not is_skll_model(new_config['model']):
                logging.warning("You specified a custom SKLL objective but also chose a "
                                "non-SKLL model. The objective will be ignored.")
            else:
                if new_config['skll_objective'] not in SCORERS:
                    raise ValueError("Invalid SKLL objective. Please refer to the SKLL "
                                     "documentation and choose a valid tuning objective.")

        # 9. Check that if we are running rsmtool to ask for
        # expected scores then the SKLL model type must actually
        # support probabilistic classification. If it's not a SKLL
        # model at all, we just treat it as a LinearRegression model
        # which is basically what they all are in the end.
        if context == 'rsmtool' and new_config['predict_expected_scores']:
            model_name = new_config['model']
            dummy_learner = Learner(model_name) if is_skll_model(model_name) else Learner('LinearRegression')
            if not hasattr(dummy_learner.model_type, 'predict_proba'):
                raise ValueError("{} does not support expected scores "
                                 "since it is not a probablistic classifier.".format(model_name))
            del dummy_learner

        # 10. Check the fields that requires rsmextra
        if not HAS_RSMEXTRA:
            if new_config['special_sections']:
                raise ValueError("Special sections are only available to ETS"
                                 " users by installing the rsmextra package.")

        # 11. Raise a warning if we are specifiying a feature file but also
        # telling the system to automatically select transformations
        if new_config['features'] and new_config['select_transformations']:
            logging.warning("You specified a feature file but also set "
                            "`select_transformations` to True. Any "
                            "transformations or signs specified in "
                            "the feature file will be overwritten by "
                            "the automatically selected transformations "
                            "and signs.")

        # 12. If we have `experiment_names`, check that the length of the list
        # matches the list of experiment_dirs.
        if context == 'rsmsummarize' and new_config['experiment_names']:
            if len(new_config['experiment_names']) != len(new_config['experiment_dirs']):
                raise ValueError("The number of specified experiment names should be the same"
                                 " as the number of specified experiment directories.")

        # 13. Check that if the user specified min_n_per_group, they also
        # specified subgroups. If they supplied a dictionary, make
        # sure the keys match
        if new_config['min_n_per_group']:
            # make sure we have subgroups
            if 'subgroups' not in new_config:
                raise ValueError("You must specify a list of subgroups in "
                                 "in the `subgroups` field if "
                                 "you want to use the `min_n_per_group` field")
            # if we got dictionary, make sure the keys match
            elif isinstance(new_config['min_n_per_group'], dict):
                if sorted(new_config['min_n_per_group'].keys()) != sorted(new_config['subgroups']):
                    raise ValueError("The keys in `min_n_per_group` must "
                                     "match the subgroups in `subgroups` field")
            # else convert to dictionary
            else:
                new_config['min_n_per_group'] = {group: new_config['min_n_per_group']
                                                 for group in new_config['subgroups']}

        # 14. Clean up config dict to keep only context-specific fields
        context_relevant_fields = (CHECK_FIELDS[context]['optional'] +
                                   CHECK_FIELDS[context]['required'])

        new_config = {k: v for k, v in new_config.items()
                      if k in context_relevant_fields}

        if inplace:
            self._config = new_config
        return Configuration(self._config,
                             configdir=self._configdir,
                             filename=self._filename,)

    def process_config(self, inplace=True):
        """
        Converts fields which are read in as string to the
        appropriate format. Fields which can take multiple
        string values are converted to lists if they have
        not been already formatted as such.

        Parameters
        ----------
        inplace : bool
            Maintain the state of the config object produced by
            this method.
            Defaults to True.

        Returns
        -------
        config_obj : Configuration
            A configuration object

        Raises
        -------
        NameError
            If config does not exist, or no config read.
        """

        # Check to make sure a configuration file
        # or dictionary has been loaded.
        self._check_config_is_loaded()

        # Get the parameter dictionary
        new_config = self._config

        # convert multiple values into lists
        for field in LIST_FIELDS:
            if field in new_config and new_config[field] is not None:
                if not isinstance(new_config[field], list):
                    new_config[field] = new_config[field].split(',')
                    new_config[field] = [prefix.strip() for prefix
                                         in new_config[field]]

        # make sure all boolean values are boolean
        for field in BOOLEAN_FIELDS:
            error_message = ('Field {} can only be set to '
                             'True or False.'.format(field))
            if field in new_config and new_config[field] is not None:
                if not isinstance(new_config[field], bool):
                    # we first convert the value to string to avoid
                    # attribute errors in case the user supplied an integer.
                    given_value = str(new_config[field]).strip()
                    m = re.match(r'^(true|false)$', given_value, re.I)
                    if not m:
                        raise ValueError(error_message)
                    else:
                        bool_value = json.loads(m.group().lower())
                        new_config[field] = bool_value

        if inplace:
            self._config = new_config
        return Configuration(self._config,
                             configdir=self._configdir,
                             filename=self._filename,)

    def normalize_validate_and_process_config(self, context='rsmtool'):
        """
        Normalize, validate, and process data from a config file.

        Parameters
        ----------
        context : str, optional
            Context of the tool in which we are validating.
            Possible values are ::

                {'rsmtool', 'rsmeval',
                 'rsmpredict', 'rsmcompare',
                 'rsmsummarize'}

            Defaults to 'rsmtool'.

        Returns
        -------
        config_obj : Configuration
            A configuration object

        Raises
        -------
        NameError
            If config does not exist, or no config read.
        """

        # Check to make sure a configuration file
        # or dictionary has been loaded.
        self._check_config_is_loaded()

        self.normalize_config()
        self.process_config()
        self.validate_config(context=context)
        return Configuration(self._config,
                             configdir=self._configdir,
                             filename=self._filename,
                             context=context)

    def read_normalize_validate_and_process_config(self,
                                                   filepath,
                                                   context='rsmtool'):
        """
        Read, normalize, validate, and process data from a config file.

        Parameters
        ----------
        filepath : str
            The path to the configuration file.
        context : str, optional
            Context of the tool in which we are validating.
            Possible values are ::

                {'rsmtool', 'rsmeval',
                 'rsmpredict', 'rsmcompare', 'rsmsummarize'}

            Defaults to 'rsmtool'.

        Returns
        -------
        config_obj : Configuration
            A configuration object
        """

        logging.info('Reading and preprocessing configuration file: {}'.format(filepath))
        self.read_config_from_file(filepath)
        return self.normalize_validate_and_process_config(context=context)


class JSONConfigurationParser(ConfigurationParser):
    """
    A subclass of `ConfigurationParser` for parsing
    JSON-style config files.
    """

    def __init__(self):

        super().__init__()

    def read_config_from_file(self, filepath):
        """
        Read the configuration file.

        Parameters
        ----------
        filepath : str
            The path to the configuration file.

        Raises
        ------
        ValueError
            If main configuration file is improperly formatted.
        """
        try:
            config = parse_json_with_comments(filepath)
        except ValueError:
            raise ValueError('The main configuration file `{}` exists but '
                             'is formatted incorrectly. Please check that '
                             'each line ends with a comma, there is no comma '
                             'at the end of the last line, and that all quotes '
                             'match.'.format(filepath))

        self._config = config
        self._filename = basename(filepath)
        self._configdir = dirname(abspath(filepath))


class CFGConfigurationParser(ConfigurationParser):
    """
    A subclass of `ConfiguraitonParser` for parsing
    Microsoft INI-style config files.
    """

    def __init__(self):

        super().__init__()

    @staticmethod
    def _fix_json(json_string):
        """
        Takes a bit of JSON that might have bad quotes
        or capitalized booleans and fixes that stuff.

        Parameters
        ----------
        json_string : str
            A string to be reformatted for JSON parsing.

        Return
        ------
        json_string : str
            The updated string.
        """
        json_string = json_string.replace('True', 'true')
        json_string = json_string.replace('False', 'false')
        json_string = json_string.replace("'", '"')
        return json_string

    def read_config_from_file(self, filepath):
        """
        Read the configuration file.

        Parameters
        ----------
        filepath : str
            The path to the configuration file.

        Raises
        ------
        ValueError
            If main configuration file is improperly formatted.
        """

        # Get the `ConfigParser` object
        py_config_parser = ConfigParser()

        # Try to read main configuration file.
        try:
            py_config_parser.read(filepath)
        except Exception as error:
            raise ValueError('Main configuration file '
                             'could not be read: {}'.format(error))

        config = {}

        # Loop through all sections of the ConfigParser
        # object and add items to the dictionary
        for section in py_config_parser.sections():
            for name, value in py_config_parser.items(section):

                # Check if the key already exists in the config dictionary.
                # If it does, skip it and log a warning.
                if name in config:
                    logging.warning('There are duplicate keys for `{}`'
                                    'in the configuration file. Only '
                                    'the first instance will be '
                                    'included.'.format(name))
                    continue

                # Otherwise, safe convert the value
                # and add it to the dictionary.
                else:

                    value = self._fix_json(value)
                    value = yaml.safe_load(value)
                    config[name] = value

        self._config = config
        self._filename = basename(filepath)
        self._configdir = dirname(abspath(filepath))


# Global config types
CONFIG_TYPE = {'.cfg': CFGConfigurationParser,
               '.json': JSONConfigurationParser}
