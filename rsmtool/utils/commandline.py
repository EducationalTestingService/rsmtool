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
import sys

from collections import namedtuple, OrderedDict
from itertools import chain, product
from pathlib import Path

from prompt_toolkit.completion import (FuzzyWordCompleter,
                                       PathCompleter,
                                       WordCompleter)
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import (clear,
                                      print_formatted_text,
                                      prompt,
                                      CompleteStyle)
from prompt_toolkit.validation import Validator

from rsmtool import VERSION_STRING
from rsmtool.configuration_parser import Configuration
from rsmtool.reporter import Reporter
from .constants import (CHECK_FIELDS,
                        CONFIGURATION_DOCUMENTATION_SLUGS,
                        DEFAULTS,
                        INTERACTIVE_MODE_METADATA,
                        POSSIBLE_EXTENSIONS)

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
                       defaults=[None, None, None, None, None, None])


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

    parser_generate.add_argument('--interactive',
                                 dest='interactive',
                                 action='store_true',
                                 default=False,
                                 help=f"if specified, generate the {name} "
                                      f"configuration file interactively")

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


class ConfigurationGenerator:
    """
    Class that encapsulates automated batch-mode and interactive
    generation of various tool configurations.

    Attributes
    ----------
    context : str
        Name of the command-line tool for which we are generating the
        configuration file.
    as_string : bool, optional
        If ``True``, return a formatted and indented string representation
        of the configuration, rather than a dictionary.
        Defaults to ``False``.
    suppress_warnings : bool, optional
        If ``True``, do not generate any warnings.
    use_subgroups : bool, optional
        If ``True``, include subgroup-related sections in the list of general sections
        in the configuration file.
        Defaults to ``False``.
    """

    def __init__(self,
                 context,
                 as_string=False,
                 suppress_warnings=False,
                 use_subgroups=False):
        self.context = context
        self.use_subgroups = use_subgroups
        self.suppress_warnings = suppress_warnings
        self.as_string = as_string
        self.logger = logging.getLogger(__name__)

        # we need to save the first required and first optional field we will
        # insert since we will use them as sign posts to insert comments later
        self._required_fields = CHECK_FIELDS[self.context]['required']
        self._optional_fields = sorted(CHECK_FIELDS[self.context]['optional'])
        self._first_required_field = self._required_fields[0]
        self._first_optional_field = self._optional_fields[0]

    def _convert_to_string(self,
                           config_object,
                           insert_url_comment=True,
                           insert_required_comment=True,
                           insert_optional_comment=True):
        configuration = str(config_object)

        # insert the URL comment first, right above the first required field
        if insert_url_comment:
            base_url = 'https://rsmtool.readthedocs.io/en/stable'
            doc_slug = CONFIGURATION_DOCUMENTATION_SLUGS[self.context]
            doc_url = f"{base_url}/{doc_slug}"
            configuration = re.sub(fr'([ ]+)("{self._first_required_field}": [^,]+,\n)',
                                   fr'\1// Reference: {doc_url}\n\1\2',
                                   configuration)

        # insert first comment right above the first required field
        if insert_required_comment:
            configuration = re.sub(fr'([ ]+)("{self._first_required_field}": [^,]+,\n)',
                                   fr'\1// REQUIRED: replace "ENTER_VALUE_HERE" with the appropriate value!\n\1\2',
                                   configuration)

        # insert second comment right above the first optional field
        if insert_optional_comment:
            configuration = re.sub(fr'([ ]+)("{self._first_optional_field}": [^,]+,\n)',
                                   r'\1// OPTIONAL: replace default values below based on your data.\n\1\2',
                                   configuration)

        return configuration

    def _expand_general_sections_list(self):

        reporter = Reporter()
        default_general_sections_value = DEFAULTS.get('general_sections', '')
        default_special_sections_value = DEFAULTS.get('special_sections', '')
        default_custom_sections_value = DEFAULTS.get('custom_sections', '')

        # if we are told ot use subgroups then just make up a dummy subgroup
        # value so that the subgroup-based sections will be included in the
        # section list. This value is not actually used in configuration file.
        subgroups_value = ['GROUP'] if self.use_subgroups else DEFAULTS.get('subgroups', '')
        return reporter.determine_chosen_sections(default_general_sections_value,
                                                  default_special_sections_value,
                                                  default_custom_sections_value,
                                                  subgroups_value,
                                                  context=self.context)

    def _make_choice_validator(self, choices):
        validator = Validator.from_callable(lambda choice: choice in choices,
                                            error_message='invalid choice')
        return validator

    def _make_file_completer(self):
        def valid_file(filename):
            return (Path(filename).is_dir() or
                    Path(filename).suffix.lower().lstrip('.') in POSSIBLE_EXTENSIONS)
        return PathCompleter(expanduser=False, file_filter=valid_file)

    def _make_file_validator(self):
        def is_valid(filepath):
            return (Path(filepath).is_file() and
                    Path(filepath).suffix.lower().lstrip('.') in POSSIBLE_EXTENSIONS)
        validator = Validator.from_callable(is_valid, error_message='invalid file')
        return validator

    def _make_file_format_validator(self):
        validator = Validator.from_callable(lambda ext: ext in POSSIBLE_EXTENSIONS or ext == '', error_message='invalid format')
        return validator

    def _make_directory_completer(self):
        return PathCompleter(expanduser=False, only_directories=True)

    def _make_directory_validator(self):
        validator = Validator.from_callable(lambda filepath: Path(filepath).is_dir(),
                                            error_message='invalid directory')
        return validator

    def _make_id_validator(self):
        validator = Validator.from_callable(lambda text: text and ' ' not in text,
                                            error_message='blanks/spaces not allowed')
        return validator

    def _make_boolean_validator(self, allow_empty=False):
        correct_choices = ['true', 'false']
        if allow_empty:
            correct_choices.append('')
        validator = Validator.from_callable(lambda answer: answer in correct_choices,
                                            error_message='invalid answer')
        return validator

    def _make_integer_validator(self, allow_empty=False):
        integer_regex = r'^[0-9]+|^$' if allow_empty else r'^[0-9]$'
        validator = Validator.from_callable(lambda answer: re.match(integer_regex, answer),
                                            error_message='invalid integer')
        return validator

    def _get_multiple_field_inputs(self, prompt_text, **kwargs):
        values = []

        print_formatted_text(HTML(f" <b>{prompt_text}</b>"))
        num_entries = prompt("  How many do you want to specify: ",
                             validator=self._make_integer_validator())
        num_entries = int(num_entries)

        for i in range(num_entries):
            value = prompt(f"   Enter #{i+1}: ", **kwargs)
            values.append(value)

        return values

    def _interactive_loop_for_field(self, field_name, field_type):
        field_metadata = INTERACTIVE_MODE_METADATA[field_name]
        prompt_label = field_metadata['prompt']
        field_data_type = field_metadata.get('type', 'text')
        field_count = field_metadata.get('count', 'single')
        possible_choices = field_metadata.get('choices', [])

        # instantiate all completers, validators, and styles as None
        field_completer = None
        field_validator = None
        complete_style = None

        # override completers, validators, and styles as necessary
        # depending on the data type of the field
        if field_data_type == 'choice':
            if not possible_choices:
                raise(ValueError, f"invalid list of choices for {field_name}")
            else:
                field_completer = FuzzyWordCompleter(possible_choices)
                field_validator = self._make_choice_validator(possible_choices)
                complete_style = CompleteStyle.MULTI_COLUMN
        elif field_data_type == 'file':
            field_completer = self._make_file_completer()
            field_validator = self._make_file_validator()
        elif field_data_type == 'format':
            field_completer = WordCompleter(POSSIBLE_EXTENSIONS)
            field_validator = self._make_file_format_validator()
        elif field_data_type == 'dir':
            field_completer = self._make_directory_completer()
            field_validator = self._make_directory_validator()
        elif field_data_type == 'id':
            field_completer = None
            field_validator = self._make_id_validator()
        elif field_data_type == 'integer':
            field_completer = None
            allow_empty = field_type == 'optional'
            field_validator = self._make_integer_validator(allow_empty=allow_empty)
        elif field_data_type == 'boolean':
            allow_empty = field_type == 'optional'
            field_completer = WordCompleter(['true', 'false'])
            field_validator = self._make_boolean_validator(allow_empty=allow_empty)

        # start the main event loop for the field
        while True:
            try:
                sys.stderr.write("\n")
                if field_count == 'multiple':
                    text = self._get_multiple_field_inputs(prompt_label,
                                                           completer=field_completer,
                                                           validator=field_validator)
                else:
                    text = prompt(HTML(f" <b>{prompt_label}</b>: "),
                                  completer=field_completer,
                                  validator=field_validator,
                                  complete_style=complete_style)
                # boolean fields need to be converted to actual booleans
                if field_data_type == 'boolean':
                    text = False if text in ['false', ''] else True
                # and integer fields to integers/None
                elif field_data_type == 'integer':
                    text = int(text) if text else None
            except KeyboardInterrupt:
                continue
            else:
                return text

    def interact(self):

        # clear the screen
        clear()

        # print the preamble and some instructions
        sys.stderr.write("\n")
        sys.stderr.write("Entering interactive mode:\n")
        sys.stderr.write(" - press ctrl-d to exit without generating a configuration\n")
        sys.stderr.write(" - press tab or start typing when choosing files/directories/models\n")
        sys.stderr.write(" - press enter to accept the default value for a field (underlined)\n")
        sys.stderr.write(" - press ctrl-c to cancel current entry for a field and enter again\n")
        sys.stderr.write("\n")

        # instantiate a blank dictionary
        configdict = OrderedDict()

        # iterate over the required fields first, and then the (sorted) optional fields
        # keep track of which field type we are currently dealing with
        for field_type, field_name in chain(product(['required'], self._required_fields),
                                            product(['optional'], self._optional_fields)):

            # set up an interactive loop for the field if appropriate
            try:
                configdict[field_name] = self._interactive_loop_for_field(field_name, field_type)
            # if the field is not one that is meant to be filled interactively,
            # then just use its default value; for "general_sections", expand it
            # so that it is easy for the user to remove sections
            except KeyError:
                if field_name == 'general_sections':
                    configdict[field_name] = self._expand_general_sections_list()
                else:
                    configdict[field_name] = DEFAULTS.get(field_name, '')
            # if the user pressed Ctrl-D, then exit out of interactive mode
            # without generating anything
            except EOFError:
                sys.stderr.write("\n")
                sys.stderr.write("You exited interactive mode without a configuration.")
                sys.stderr.write("\n")
                return ''

        sys.stderr.write("\n")
        config_object = Configuration(configdict,
                                      filename=f"example_{self.context}.json",
                                      configdir=os.getcwd(),
                                      context=self.context)
        return self._convert_to_string(config_object, insert_required_comment=False)

    def generate(self):
        """
        Automatically generate an example configuration in batch mode.

        Returns
        -------
        configuration : dict or str
            The generated configuration either as a dictionary or
            a formatted string, depending on the value of ``as_string``.
        """
        # instantiate a dictionary that remembers key insertion order
        configdict = OrderedDict()

        # insert the required fields first and give them a dummy value
        for required_field in self._required_fields:
            configdict[required_field] = 'ENTER_VALUE_HERE'

        # insert the optional fields in alphabetical order
        for optional_field in self._optional_fields:

            # to make it easy for users to add/remove sections, we should
            # populate the `general_sections` field with an explicit list
            # instead of the default value which is simply ``['all']``. To
            # do this, we can use the reporter class.
            if optional_field == 'general_sections':
                configdict['general_sections'] = self._expand_general_sections_list()
            else:
                configdict[optional_field] = DEFAULTS.get(optional_field, '')

        # create a Configuration object
        config_object = Configuration(configdict,
                                      filename=f"example_{self.context}.json",
                                      configdir=os.getcwd(),
                                      context=self.context)

        # if we were asked for string output, then convert this dictionary to
        # a string that will also insert some useful comments
        if self.as_string:
            configuration = self._convert_to_string(config_object)
        # otherwise we just return the dictionary underlying the Configuration object
        else:
            configuration = config_object._config

        # print out a warning to make it clear that it cannot be used as is
        if not self.suppress_warnings:
            self.logger.warning("Automatically generated configuration files MUST "
                                "be edited to add values for required fields and "
                                "even for optional ones depending on your data.")

        # return either the Configuration object or the string
        return configuration
