"""
Utility functions used in RSMTool command-line tools.

:author: Nitin Madnani (nmadnani@ets.org)
:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)

:organization: ETS
"""

import argparse
import logging
import os
import re

from collections import namedtuple, OrderedDict
from pathlib import Path

from rsmtool import VERSION_STRING
from rsmtool.configuration_parser import Configuration
from rsmtool.reporter import Reporter
from .constants import CHECK_FIELDS, CONFIGURATION_DOCUMENTATION_SLUGS, DEFAULTS

# a named tuple for use with the `setup_rsmcmd_parser` function below
# to specify additional options for either of the subcommand parsers.
# An example can be found in `rsmpredict.py`. All of the attributes
# are directly named for the arguments that are used with
# the `ArgParser.add_argument()` method. The `dest` and `help`
# options are required but the rest can be left unspecified and
# will default to `None`.
CmdOption = namedtuple('CmdOption',
                       ['dest', 'help', 'shortname', 'longname', 'action',
                        'default', 'required', 'nargs'])
# if rsmtool were python 3.7+ only, we could just use the `defaults`
# keyword argument to specify the default values; but to support python
# 3.6, we need to mess with the `__new__` constructor.
# Adapted from: https://stackoverflow.com/a/18348004
# TODO: replace this with `defaults` when we drop support for python 3.7
CmdOption.__new__.__defaults__ = (None,) * 6

def setup_rsmcmd_parser(name,
                        uses_output_directory=True,
                        allows_overwriting=False,
                        extra_run_options=[],
                        uses_subgroups=False):
    """
    A helper function to create argument parsers for RSM command-line utilities.

    Since the various RSM command-line utilities (``rsmtool``, ``rsmeval``,
    ``rsmcompare``, etc.) have very similar argument parsers, refactoring that
    shared code out into this helper function makes it easier to extend
    and modify the command-line interface to all utilities at the same time.
    In addition, it also improves the consistency among the tools.

    By default, this function adds the following options to the parser:
      - ``config_file`` : a positional argument for the tool's configuration file
      - ``-V``/``--version`` : an optional argument to print out the package version

    If ``uses_output_directory`` is ``True``, an ``output_dir`` positional
    argument will be added to the "run" subcommand parser.

    If ``allows_overwriting`` is ``True``, an ``-f``/``--force``
    optional argument will be added to the "run" subcommand parser.

    The ``extra_run_options`` list should contain a list of ``CmdOption``
    instances which are added to the "run" subcommand parser one by one.

    If ``uses_subgroups`` is ``True``, a ``--subgroups`` optional
    argument will be added to the "generate" subcommand parser.

    Parameters
    ----------
    name : str
        The name of the command-line tool for which we need the parser.
    uses_output_directory : bool, optional
        Add the ``output_dir`` positional argument to the "run" subcommand
        parser. This argument means that the respective tool uses an output
        directory to store its various outputs.
    allows_overwriting : bool, optional
        Add the ``-f``/``-force_write`` optional argument to the "run" subcommand
        parser. This argument allows the output for the respective
        tool to be overwritten even if it already exists (file) or contains
        output (directory).
    extra_run_options : list, optional
        Any additional options to be added to the "run" subcommand parser,
        each specified as a ``CmdOption`` instance.
    uses_subgroups : bool, optional
        Add the ``--subgroups`` optional argument to the "generate" subcommand
        parser. This argument means that the tool for which we are automatically
        generating a configuration file includes additional information when
        subgroup information is available.

    Returns
    -------
    parser : arpgarse.ArgumentParser
        A fully instantiated argument parser for the respective tool.

    Raises
    ------
    RuntimeError
        If any of the ``CmdOption`` instances specified in
        ``extra_run_options`` do not contain the ``dest`` and
        ``help`` attributes.

    Note
    ----
    This function is only meant to be used by RSMTool developers.
    """

    # a special callable to test whether configuration files exist
    # or not; this is nested because it is only used within this function
    # and should never be used externally
    def existing_configuration_file(string):
        if Path(string).exists():
            return string
        else:
            msg = 'The configuration file %r does not exist.' % string
            raise argparse.ArgumentTypeError(msg)

    # initialize an argument parser
    parser = argparse.ArgumentParser(prog=f"{name}")

    # we always want to have a version flag for the main parser
    parser.add_argument('-V',
                        '--version',
                        action='version',
                        version=VERSION_STRING,
                        help=f"show the {name} version number and exit")

    # each RSM command-line utility has two subcommands
    # - generate : used to auto-generate configuration files
    # - run : used to run experiments

    # let's set up the sub-parsers corresponding to these subcommands
    subparsers = parser.add_subparsers(dest='subcommand', title='subcommands')
    parser_generate = subparsers.add_parser('generate',
                                            help=f"automatically generate an "
                                                 f"{name} configuration file")
    parser_run = subparsers.add_parser('run',
                                       help=f"run an {name} experiment")

    ###################################################
    # Setting up options for the "generate" subparser #
    ###################################################
    if uses_subgroups:
        parser_generate.add_argument('--subgroups',
                                     dest='subgroups',
                                     action='store_true',
                                     default=False,
                                     help=f"if specified, the generated {name} "
                                          f"configuration file will include the "
                                          f"subgroup sections in the general "
                                          f"sections list")

    parser_generate.add_argument('-q',
                                 '--quiet',
                                 dest='quiet',
                                 action='store_true',
                                 default=False,
                                 help="if specified, the warning about not "
                                      "using the generated configuration "
                                      "as-is will be suppressed.")

    ##############################################
    # Setting up options for the "run" subparser #
    ##############################################

    # since this is an RSMTool command-line utility, we will
    # always need a configuration file
    parser_run.add_argument('config_file',
                            type=existing_configuration_file,
                            help=f"the {name} JSON configuration file to run")

    # if it uses an output directory, let's add that
    if uses_output_directory:
        parser_run.add_argument('output_dir',
                                nargs='?',
                                default=os.getcwd(),
                                help="the output directory where all the files "
                                     "for this run will be stored")

    # if it allows overwrting the output directory, let's add that
    if allows_overwriting:
        parser_run.add_argument('-f',
                                '--force',
                                dest='force_write',
                                action='store_true',
                                default=False,
                                help=f"if specified, {name} will overwrite the "
                                     f"contents of the output file or directory "
                                     f"even if it contains the output of a "
                                     f"previous run ")

    # add any extra options passed in for the rub subcommand;
    for parser_option in extra_run_options:

        # construct the arguments and keyword arguments needed for the
        # `add_argument()` call to the parser
        argparse_option_args = []
        argparse_option_kwargs = {}

        # first add the destination and the help string
        argparse_option_kwargs["dest"] = f"{parser_option.dest}"
        argparse_option_kwargs["help"] = f"{parser_option.help}"

        # now add any optional information
        if parser_option.shortname is not None:
            argparse_option_args.append(f"-{parser_option.shortname}")
        if parser_option.longname is not None:
            argparse_option_args.append(f"--{parser_option.longname}")
        if parser_option.action is not None:
            argparse_option_kwargs['action'] = f"{parser_option.action}"
        if parser_option.default is not None:
            argparse_option_kwargs["default"] = f"{parser_option.default}"
        if parser_option.required is not None:
            try:
                assert type(parser_option.required) == bool
            except AssertionError:
                raise TypeError(f"the 'required' field for CmdOption must be "
                                f"boolean, you specified '{parser_option.required}'")
            else:
                argparse_option_kwargs["required"] = parser_option.required
        if parser_option.nargs is not None:
            argparse_option_kwargs['nargs'] = f"{parser_option.nargs}"

        # add this argument to the parser
        parser_run.add_argument(*argparse_option_args, **argparse_option_kwargs)

    return parser


def generate_configuration(context,
                           use_subgroups=False,
                           as_string=False,
                           suppress_warnings=False):
    """
    Automatically generate an example configuration for a given
    command-line tool.

    Parameters
    ----------
    context : str
        Name of the command-line tool for which we are generating the
        configuration file.
    use_subgroups : bool, optional
        If ``True``, include subgroup-related sections in the list of general sections
        in the configuration file.
        Defaults to ``False``.
    as_string : bool, optional
        If ``True``, return a formatted and indented string representation
        of the configuration, rather than a dictionary.
        Defaults to ``False``.
    suppress_warnings : bool, optional
        If ``True``, do not generate any warnings.

    Returns
    -------
    configuration : dict or str
        The generated configuration either as a dictionary or
        a formatted string, depending on the value of ``as_string``.
    """
    # get a logger for this function
    logger = logging.getLogger(__name__)

    # get the fields we are going to put in the config file
    required_fields = CHECK_FIELDS[context]['required']
    optional_fields = CHECK_FIELDS[context]['optional']

    # the optional fields will be inserted in alphabetical order
    sorted_optional_fields = sorted(optional_fields)

    # we need to save the first required and first optional field we will
    # insert since we will use them as sign posts to insert comments later
    first_required_field = required_fields[0]
    first_optional_field = sorted_optional_fields[0]

    # instantiate a dictionary that remembers key insertion order
    configdict = OrderedDict()

    # insert the required fields first and give them a dummy value
    for required_field in required_fields:
        configdict[required_field] = 'ENTER_VALUE_HERE'

    # insert the optional fields in alphabetical order
    for optional_field in sorted_optional_fields:

        # to make it easy for users to add/remove sections, we should
        # populate the `general_sections` field with an explicit list
        # instead of the default value which is simply ``['all']``. To
        # do this, we can use the reporter class.
        if optional_field == 'general_sections':
            reporter = Reporter()
            default_general_sections_value = DEFAULTS.get('general_sections', '')
            default_special_sections_value = DEFAULTS.get('special_sections', '')
            default_custom_sections_value = DEFAULTS.get('custom_sections', '')

            # if we are told ot use subgroups then just make up a dummy subgroup
            # value so that the subgroup-based sections will be included in the
            # section list. This value is not actually used in configuration file.
            subgroups_value = ['GROUP'] if use_subgroups else DEFAULTS.get('subgroups', '')
            configdict['general_sections'] = reporter.determine_chosen_sections(
                default_general_sections_value,
                default_special_sections_value,
                default_custom_sections_value,
                subgroups_value,
                context=context)
        else:
            configdict[optional_field] = DEFAULTS.get(optional_field, '')

    # create a Configuration object
    config_object = Configuration(configdict,
                                  filename=f"example_{context}.json",
                                  configdir=os.getcwd(),
                                  context=f"{context}")

    # if we were asked for string output, then convert the Configuration
    # object to a string and also insert some useful comments to print out
    if as_string:

        configuration = str(config_object)

        # insert first comment right above the first required field
        base_url = 'https://rsmtool.readthedocs.io/en/stable'
        doc_slug = CONFIGURATION_DOCUMENTATION_SLUGS[context]
        doc_url = f"{base_url}/{doc_slug}"
        configuration = re.sub(fr'([ ]+)("{first_required_field}": [^,]+,\n)',
                               fr'\1// REQUIRED: replace "ENTER_VALUE_HERE" with the appropriate value!\n\1// {doc_url}\n\1\2',
                               configuration)

        # insert second comment right above the first optional field
        configuration = re.sub(fr'([ ]+)("{first_optional_field}": [^,]+,\n)',
                               r'\1// OPTIONAL: replace default values below based on your data.\n\1\2',
                               configuration)
    # otherwise we just return the dictionary underlying the Configuration object
    else:
        configuration = config_object._config

    # print out a warning to make it clear that it cannot be used as is
    if not suppress_warnings:
        logger.warning("Automatically generated configuration files MUST "
                       "be edited to add values for required fields and "
                       "even for optional ones depending on your data.")

    # return either the Configuration object or the string
    return configuration
