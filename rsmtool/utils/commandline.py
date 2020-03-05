"""
Utility functions specific for command-line tools.

:author: Nitin Madnani (nmadnani@ets.org)
:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)

:organization: ETS
"""

import argparse
import json
import logging
import os
from collections import namedtuple, OrderedDict
from pathlib import Path

from rsmtool import VERSION_STRING
from rsmtool.configuration_parser import Configuration
from rsmtool.reporter import Reporter

# a named tuple for use with the `setup_rsmcmd_parser` function below
# to specify additional options for either of the subcommand parsers.
# An example can be found in `rsmpredict.py`. All of the attributes
# are directly named for the arguments that are used with
# the `ArgParser.add_argument()` method. The `dest` and `help`
# options are required but the rest can be left unspecified and
# will default to `None`.
CmdOption = namedtuple('CmdOption',
                       ['dest', 'help', 'shortname', 'longname', 'action',
                        'default', 'required', 'nargs'],
                       defaults=[None, None, None, None, None])


def setup_rsmcmd_parser(name,
                        uses_output_directory=True,
                        allows_overwriting_directory=False,
                        extra_run_options=[],
                        with_subgroup_sections=False):
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

    If ``allows_overwriting_directory`` is ``True``, an ``-f``/``--force``
    optional argument will be added to the "run" subcommand parser.

    The ``extra_run_options`` list should contain a list of ``CmdOption``
    instances which are added to the "run" subcommand parser one by one.

    If ``with_subgroup_sections`` is ``True``, a ``--groups`` optional
    argument will be added to the "quickstart" subcommand parser.

    Parameters
    ----------
    name : str
        The name of the command-line tool for which we need the parser.
    uses_output_directory : bool, optional
        Add the ``output_dir`` positional argument to the "run" subcommand
        parser. This argument means that the respective tool uses an output
        directory to store its various outputs.
    allows_overwriting_directory : bool, optional
        Add the ``-f``/``-force_write`` optional argument to the "run" subcommand
        parser. This argument allows the output directory for the respective
        tool to be overwritten even if it already contains some output.
    extra_run_options : list, optional
        Any additional options to be added to the "run" subcommand parser,
        each specified as a ``CmdOption`` instance.
    with_subgroup_sections : bool, optional
        Add the ``--groups`` optional argument to the "quickstart" subcommand
        parser.

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
    parser.add_argument('-V', '--version', action='version', version=VERSION_STRING)

    # each RSM command-line utility has two subcommands
    # - quickstart : used to auto-generate configuration files
    # - run : used to run experiments

    # let's set up the sub-parsers corresponding to these subcommands
    subparsers = parser.add_subparsers(dest='subcommand')
    parser_quickstart = subparsers.add_parser('quickstart',
                                              help=f"Automatically generate an {name} configuration file")
    parser_run = subparsers.add_parser('run',
                                       help=f"Run an {name} experiment")

    #####################################################
    # Setting up options for the "quickstart" subparser #
    #####################################################
    if with_subgroup_sections:
        parser_quickstart.add_argument('--groups',
                                       dest='subgroups',
                                       action='store_true',
                                       default=False,
                                       help=f"If true, the generated {name} "
                                            "configuration file will include the "
                                            "subgroup sections in the general "
                                            "sections list.")

    ##############################################
    # Setting up options for the "run" subparser #
    ##############################################

    # since this is an RSMTool command-line utility, we will
    # always need a configuration file
    parser_run.add_argument('config_file',
                            type=existing_configuration_file,
                            help="The JSON configuration file for this experiment")

    # if it uses an output directory, let's add that
    if uses_output_directory:
        parser_run.add_argument('output_dir',
                                nargs='?',
                                default=os.getcwd(),
                                help="The output directory where all the files "
                                     "for this experiment will be stored")

    # if it allows overwrting the output directory, let's add that
    if allows_overwriting_directory:
        parser_run.add_argument('-f',
                                '--force',
                                dest='force_write',
                                action='store_true',
                                default=False,
                                help=f"If true, {name} will not check if the "
                                     "output directory already contains the "
                                     "output of another {name} experiment. ")

    # add any extra options passed in for the rub subcommand; each of them must
    # have a destination name and a help string
    for parser_option in extra_run_options:
        try:
            assert hasattr(parser_option, 'dest')
            assert hasattr(parser_option, 'help')
        except AssertionError:
            # this exception should _never_ be encountered by a user
            # this function is really only meant for RSMTool developers
            raise RuntimeError(f"Invalid option {parser_option} for {name} parser.")
        else:
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
                argparse_option_kwargs['action'] = f"'{parser_option.action}'"
            if parser_option.default is not None:
                argparse_option_kwargs["default"] = f"{parser_option.default}"
            if parser_option.required is not None:
                argparse_option_kwargs["required"] = f"{parser_option.required}"
            if parser_option.nargs is not None:
                argparse_option_kwargs['nargs'] = f"'{parser_option.nargs}'"

            # add this argument to the parser
            parser_run.add_argument(*argparse_option_args, **argparse_option_kwargs)

    return parser


def generate_configuration(name,
                           interactive=False,
                           use_subgroups=False,
                           as_string=False):

    # get a logger for this function
    logger = logging.getLogger(__name__)

    # get the fields we are going ot put in the config file
    required_fields = CHECK_FIELDS[name]['required']
    optional_fields = CHECK_FIELDS[name]['optional']

    # instantiate a dictionary that remembers key insertion order
    configdict = OrderedDict()

    # add a dummy field that will be replaced with a comment about required fields
    configdict['comment1'] = 'REQUIRED_FIELDS_COMMENT'

    # insert the required fields first and give them a dummy value
    for required_field in required_fields:
        configdict[required_field] = 'ENTER_VALUE_HERE'

    # add a dummy field that will be replaced with a comment about optional fields
    configdict['comment2'] = 'OPTIONAL_FIELDS_COMMENT'

    # insert the optional fields in alphabetical order
    for optional_field in sorted(optional_fields):

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
                context=name)
        else:
            configdict[optional_field] = DEFAULTS.get(optional_field, '')

    # create a Configuration object
    configuration = Configuration(configdict,
                                  filename=f"example_{name}.json",
                                  context=f"{name}")

    json_string = json.dumps(configdict, indent=4)

    # replace the two dummy fields with actual comments
    json_string = json_string.replace('"comment1": "REQUIRED_FIELDS_COMMENT",',
                                      '// REQUIRED: replace "ENTER_VALUE_HERE" with the appropriate value!')
    json_string = json_string.replace('"comment2": "OPTIONAL_FIELDS_COMMENT",',
                                      '// OPTIONAL: replace default values below based on your data.')

    # print out a warning to make it clear that it cannot be used as is
    logger.warning("Automatically generated configuration files MUST "
                   "be edited to add values for required fields and "
                   "even for optional ones depending on your data.")

    # return the JSON string
    return json_string


class LogFormatter(logging.Formatter):
    """
    Custom logging formatter.

    Adapted from:
        http://stackoverflow.com/questions/1343227/
        can-pythons-logging-format-be-modified-depending-
        on-the-message-log-level
    """

    info_fmt = "%(msg)s"
    warn_fmt = "WARNING: %(msg)s"

    err_fmt = "ERROR: %(msg)s"
    dbg_fmt = "DEBUG: %(module)s: %(lineno)d: %(msg)s"

    def __init__(self, fmt="%(levelno)s: %(msg)s"):

        logging.Formatter.__init__(self, fmt)

    def format(self, record):
        """
        format the logger

        Parameters
        ----------
        record
            The record to format
        """

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._fmt

        # Replace the original format with one customized by logging level
        if record.levelno == logging.DEBUG:
            self._fmt = LogFormatter.dbg_fmt
            self._style = logging.PercentStyle(self._fmt)

        elif record.levelno == logging.WARNING:
            self._fmt = LogFormatter.warn_fmt
            self._style = logging.PercentStyle(self._fmt)

        elif record.levelno == logging.INFO:
            self._fmt = LogFormatter.info_fmt
            self._style = logging.PercentStyle(self._fmt)

        elif record.levelno == logging.ERROR:
            self._fmt = LogFormatter.err_fmt
            self._style = logging.PercentStyle(self._fmt)

        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._fmt = format_orig

        return result
