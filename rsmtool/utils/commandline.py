"""
Utility functions for RSMTool command-line tools.

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
from collections import OrderedDict, namedtuple
from itertools import chain, product
from pathlib import Path

from prompt_toolkit.completion import FuzzyWordCompleter, PathCompleter, WordCompleter
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.shortcuts import CompleteStyle, clear, print_formatted_text, prompt
from prompt_toolkit.validation import Validator

from rsmtool import VERSION_STRING
from rsmtool.configuration_parser import Configuration
from rsmtool.reporter import Reporter

from .constants import (
    CHECK_FIELDS,
    CONFIGURATION_DOCUMENTATION_SLUGS,
    DEFAULTS,
    INTERACTIVE_MODE_METADATA,
    POSSIBLE_EXTENSIONS,
)

# a named tuple for use with the `setup_rsmcmd_parser` function below
# to specify additional options for either of the subcommand parsers.
# An example can be found in `rsmpredict.py`. All of the attributes
# are directly named for the arguments that are used with
# the `ArgParser.add_argument()` method. The `dest` and `help`
# options are required but the rest can be left unspecified and
# will default to `None`.
CmdOption = namedtuple(
    "CmdOption",
    ["dest", "help", "shortname", "longname", "action", "default", "required", "nargs"],
    defaults=(None,) * 6,
)


def setup_rsmcmd_parser(
    name,
    uses_output_directory=True,
    allows_overwriting=False,
    extra_run_options=[],
    uses_subgroups=False,
):
    """
    Create argument parsers for RSM command-line utilities.

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
        Defaults to ``True``.
    allows_overwriting : bool, optional
        Add the ``-f``/``-force_write`` optional argument to the "run" subcommand
        parser. This argument allows the output for the respective
        tool to be overwritten even if it already exists (file) or contains
        output (directory).
        Defaults to ``False``.
    extra_run_options : list, optional
        Any additional options to be added to the "run" subcommand parser,
        each specified as a ``CmdOption`` instance.
        Defaults to ``[]``.
    uses_subgroups : bool, optional
        Add the ``--subgroups`` optional argument to the "generate" subcommand
        parser. This argument means that the tool for which we are automatically
        generating a configuration file includes additional information when
        subgroup information is available.
        Defaults to ``False``.

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
            msg = f"The configuration file {string!r} does not exist."
            raise argparse.ArgumentTypeError(msg)

    # initialize an argument parser
    parser = argparse.ArgumentParser(prog=f"{name}")

    # we always want to have a version flag for the main parser
    parser.add_argument(
        "-V",
        "--version",
        action="version",
        version=VERSION_STRING,
        help=f"show the {name} version number and exit",
    )

    # each RSM command-line utility has two subcommands
    # - generate : used to auto-generate configuration files
    # - run : used to run experiments

    # let's set up the sub-parsers corresponding to these subcommands
    subparsers = parser.add_subparsers(dest="subcommand", title="subcommands")
    parser_generate = subparsers.add_parser(
        "generate", help=f"automatically generate an " f"{name} configuration file"
    )
    parser_run = subparsers.add_parser("run", help=f"run an {name} experiment")

    ###################################################
    # Setting up options for the "generate" subparser #
    ###################################################
    if uses_subgroups:
        # we need to display a special help message for ``rsmxval``
        # since its config does not actually contain a sections list
        if name == "rsmxval":
            parser_generate.add_argument(
                "-g",
                "--subgroups",
                dest="subgroups",
                action="store_true",
                default=False,
                help=f"if specified, {name} will ensure that "
                f"subgroup sections are included in "
                f"the various reports",
            )
        else:
            parser_generate.add_argument(
                "-g",
                "--subgroups",
                dest="subgroups",
                action="store_true",
                default=False,
                help=f"if specified, the generated {name} "
                f"configuration file will include the "
                f"subgroup sections in the general "
                f"sections list",
            )

    parser_generate.add_argument(
        "-o",
        "--output",
        dest="output_file",
        type=argparse.FileType("w", encoding="utf-8"),
        required=False,
        default=None,
        help=f"if specified, the generated {name} configuration will be "
        f"written out to this file.",
    )

    parser_generate.add_argument(
        "-q",
        "--quiet",
        dest="quiet",
        action="store_true",
        default=False,
        help="if specified, the warning about not "
        "using the generated configuration "
        "as-is will be suppressed.",
    )

    parser_generate.add_argument(
        "-i",
        "--interactive",
        dest="interactive",
        action="store_true",
        default=False,
        help=f"if specified, generate the {name} configuration file interactively",
    )

    ##############################################
    # Setting up options for the "run" subparser #
    ##############################################

    # since this is an RSMTool command-line utility, we will
    # always need a configuration file
    parser_run.add_argument(
        "config_file",
        type=existing_configuration_file,
        help=f"the {name} JSON configuration file to run",
    )

    # if it uses an output directory, let's add that
    if uses_output_directory:
        parser_run.add_argument(
            "output_dir",
            nargs="?",
            default=os.getcwd(),
            help="the output directory where all the files " "for this run will be stored",
        )

    # if it allows overwrting the output directory, let's add that
    if allows_overwriting:
        parser_run.add_argument(
            "-f",
            "--force",
            dest="force_write",
            action="store_true",
            default=False,
            help=f"if specified, {name} will overwrite the "
            f"contents of the output file or directory "
            f"even if it contains the output of a "
            f"previous run ",
        )

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
            argparse_option_kwargs["action"] = f"{parser_option.action}"
        if parser_option.default is not None:
            argparse_option_kwargs["default"] = f"{parser_option.default}"
        if parser_option.required is not None:
            try:
                assert type(parser_option.required) == bool
            except AssertionError:
                raise TypeError(
                    f"the 'required' field for CmdOption must be "
                    f"boolean, you specified '{parser_option.required}'"
                )
            else:
                argparse_option_kwargs["required"] = parser_option.required
        if parser_option.nargs is not None:
            argparse_option_kwargs["nargs"] = f"{parser_option.nargs}"

        # add this argument to the parser
        parser_run.add_argument(*argparse_option_args, **argparse_option_kwargs)

    return parser


class InteractiveField:
    """
    Class that encapsulates a configuration field that is computed interactively.

    Attributes
    ----------
    choices : list
        List of possible choices for the field value if ``data_type`` is "choice".
        An empty list for all other types of fields.
    complete_style : prompt_toolkit.shortcuts.prompt.CompleteStyle
        A CompleteStyle that defines how to style the completions.
        Set to ``CompleteStyle.MULTI_COLUMN`` for fields that have ``data_type``
        of "choice". Set to ``None`` for all other field types.
    completer : prompt_toolkit.completions.base.Completer
        A ``Completer`` object used to provide auto-completion for the field.
        The actual completer used depends on the ``data_type`` of the field.
        For example, for fields of type "choice", we use a ``FuzzyWordCompleter``,
        and for fields of type "dir" and "file, we use a ``PathCompleter``.
    count : str
        An attribute indicating whether the field accepts a "single" value
        or "multiple" values. For fields that require multiple values, e.g.
        subgroups, a different strategy is used for interactive display.
    data_type : str
        A string indicating the data type of the field.
        One of the following:
        - "boolean" : a field that only accepts True/False
        - "choice" : a field that accepts one out of a fixed list of values.
        - "dir" : a field that accepts a path to a directory.
        - "file" : a field that accepts a path to a file
        - "format" : a field that accepts possible intermediate file
          formats ("csv", "tsv", and "xlsx")
        - "id" : a field that accepts experiment/comarison/summary IDs
        - "integer" : a field that accepts integer values
        - "text" : a field that accepts open-ended text
    label : str
        The label for the field that will be displayed to the user.
    prompt_method : callable
        The function that will be used to compute the value for the field.
        The main difference arises between fields that accept only a single
        value vs. multiple values.
    validator : prompt_toolkit.validation.Validator
        A ``Validator`` object used to validate the values for the field
        as they are entered by the user. Just like ``completer``, the
        type of ``Validator`` used depends on the ``data_type`` of the field.
        For example,
    """

    def __init__(self, field_name, field_type, field_metadata):
        """
        Create a new InteractiveField instance.

        Create a new instance for the given field name and with the given
        field type ("required" or "optional").

        Parameters
        ----------
        field_name : str
            The internal name of the field as used in the configuration
            dictionary.
        field_type : str
            One of "required" or "optional", depending on whether the
            field is required or optional in the configuration.
        field_metadata : dict
            A dictionary containing the pre-defined metadata attributes
            for the given field. This dictionary is required to have
            the "label" key and can have the following optional
            keys: "choices", "count", and "type". For descriptions of what
            these keys mean, see the docstring for the ``InteractiveField``
            class. Examples of such dictionaries can be found in
            ``rsmtool.utils.constants.INTERACTIVE_MODE_METADATA``.

        Raises
        ------
        ValueError
            If the list of choices is not available for a field
            of type "choice".
        """
        # assign metadata attributes to class attributes
        self.field_name = field_name
        self.field_type = field_type
        self.label = field_metadata["label"]
        self.choices = field_metadata.get("choices", [])
        self.count = field_metadata.get("count", "single")
        self.data_type = field_metadata.get("type", "text")

        # instantiate the interaction-related attributes to their default values
        self.completer = None
        self.complete_style = None
        self.validator = None

        # now override these attributes as necessary depending on field data types
        if self.data_type == "boolean":
            allow_empty = field_type == "optional"
            self.completer = WordCompleter(["true", "false"])
            self.validator = self._make_boolean_validator(allow_empty=allow_empty)
        elif self.data_type == "choice":
            if not self.choices:
                raise ValueError(f"invalid list of choices for field '{field_name}'")
            else:
                self.completer = FuzzyWordCompleter(self.choices)
                self.validator = self._make_choice_validator(self.choices)
                self.complete_style = CompleteStyle.MULTI_COLUMN
        elif self.data_type == "dir":
            self.completer = self._make_directory_completer()
            self.validator = self._make_directory_validator()
        elif self.data_type == "file":
            allow_empty = field_type == "optional"
            self.completer = self._make_file_completer()
            self.validator = self._make_file_validator(allow_empty=allow_empty)
        elif self.data_type == "format":
            self.completer = WordCompleter(POSSIBLE_EXTENSIONS)
            self.validator = self._make_file_format_validator()
        elif self.data_type == "id":
            self.completer = None
            self.validator = self._make_id_validator()
        elif self.data_type == "integer":
            self.completer = None
            allow_empty = field_type == "optional"
            self.validator = self._make_integer_validator(allow_empty=allow_empty)

    def _make_boolean_validator(self, allow_empty=False):
        """
        Create a validator for boolean fields.

        This private method creates a validator for a field with
        ``data_type`` of "boolean".

        Parameters
        ----------
        allow_empty : bool, optional
            If ``True``, it will allow the user to also just press
            enter (i.e., input a blank string), in addition to "true"
            or "false".
            Defaults to ``False``.

        Returns
        -------
        validator : prompt_toolkit.validation.Validator
            A ``Validator`` instance that ensures that the user
            input for the field is "true" / "false", or possibly
            the empty string, if ``allow_empty`` is ``True``.
        """
        correct_choices = ["true", "false"]
        if allow_empty:
            correct_choices.append("")
        validator = Validator.from_callable(
            lambda answer: answer in correct_choices, error_message="invalid answer"
        )
        return validator

    def _make_choice_validator(self, choices):
        """
        Create a validator for choice fields.

        This private method creates a validator for a field
        with ``data_type`` of "choice".

        Parameters
        ----------
        choices : list
            List of possible values for the field.

        Returns
        -------
        validator : prompt_toolkit.validation.Validator
            A ``Validator`` instance that ensures that the user
            input for the field is one of the possible choices.
        """
        validator = Validator.from_callable(
            lambda choice: choice in choices, error_message="invalid choice"
        )
        return validator

    def _make_directory_completer(self):
        """
        Create a completer for directory fields.

        This private method creates a completer for a field
        with ``data_type`` of "dir".

        Returns
        -------
        completer : prompt_toolkit.completion.base.Completer
            A ``Completer`` instance that suggests directory names
            as potential completions for user input.
        """
        return PathCompleter(expanduser=False, only_directories=True)

    def _make_directory_validator(self):
        """
        Create a validator for directory fields.

        This private method creates a validator for a field
        with ``data_type`` of "dir".

        Returns
        -------
        validator : prompt_toolkit.validation.Validator
            A ``Validator`` instance that makes sure that only
            directory names are chosen as the final user input.
        """
        validator = Validator.from_callable(
            lambda filepath: Path(filepath).is_dir(), error_message="invalid directory"
        )
        return validator

    def _make_file_completer(self):
        """
        Create a completer for file fields.

        This private method creates a completer for a field
        with ``data_type`` of "file".

        Returns
        -------
        completer : prompt_toolkit.completion.base.Completer
            A ``Completer`` instance that suggests directory names
            and files with valid input file extensions as potential
            completions for user input. Valid input file
            extensions are "csv", "jsonlines", "sas7bdat", "tsv",
            and "xlsx". We need directory names so that
            users can look into sub-directories etc.
        """

        def valid_file(filename):
            return Path(filename).is_dir() or Path(filename).suffix.lower().lstrip(".") in [
                "csv",
                "jsonlines",
                "sas7bdat",
                "tsv",
                "xlsx",
            ]

        return PathCompleter(expanduser=False, file_filter=valid_file)

    def _make_file_validator(self, allow_empty=False):
        """
        Create a validator for file fields.

        This private method creates a validator for a field
        with ``data_type`` of "file".

        Parameters
        ----------
        allow_empty : bool, optional
            If ``True``, it will allow the user to also just press
            enter (i.e., input a blank string)
            Defaults to ``False``.

        Returns
        -------
        validator : prompt_toolkit.validation.Validator
            A ``Validator`` instance that makes sure that only
            actually existing files with valid input file extensions
            are chosen as the final user input. Valid input file
            extensions are "csv", "jsonlines", "sas7bdat", "tsv",
            and "xlsx".
        """

        def is_valid(path):
            return Path(path).is_file() and Path(path).suffix.lower().lstrip(".") in [
                "csv",
                "jsonlines",
                "sas7bdat",
                "tsv",
                "xlsx",
            ]

        def is_empty(path):
            return path == ""

        validator = Validator.from_callable(
            lambda path: is_valid(path) or (allow_empty and is_empty(path)),
            error_message="invalid file",
        )
        return validator

    def _make_file_format_validator(self):
        """
        Create a validator for file format fields.

        This private method creates a validator for a field
        with ``data_type`` of "format".

        Returns
        -------
        validator : prompt_toolkit.validation.Validator
            A ``Validator`` instance that makes sure that only
            valid intermediate file extensions ("csv", "tsv",
            and "xlsx") and empty string are allowed as final
            user input. We want to allow empty string because
            intermediate file formats are optional to specify.
        """
        validator = Validator.from_callable(
            lambda ext: ext in POSSIBLE_EXTENSIONS or ext == "",
            error_message="invalid format",
        )
        return validator

    def _make_id_validator(self):
        """
        Create a validator for id fields.

        This private method creates a validator for a field
        with ``data_type`` of "id".

        Returns
        -------
        validator : prompt_toolkit.validation.Validator
            A ``Validator`` instance that makes sure that IDs
            specified by the user are not blank and do not
            contain spaces. We do not allow blanks since IDs
            are always required.
        """
        validator = Validator.from_callable(
            lambda text: len(text) > 0 and " " not in text,
            error_message="blanks/spaces not allowed",
        )
        return validator

    def _make_integer_validator(self, allow_empty=False):
        """
        Create a validator for integer fields.

        This private method creates a validator for a field
        with ``data_type`` of "integers".

        Parameters
        ----------
        allow_empty : bool, optional
            If ``True``, it will allow the user to also just press
            enter (i.e., input a blank string)
            Defaults to ``False``.

        Returns
        -------
        validator : prompt_toolkit.validation.Validator
            A ``Validator`` instance that makes sure that the
            final user input is a string representation of a
            fixed-point number or integer. Blank strings may
            also be allowed if ``allow_empty`` is ``True``.
        """
        integer_regex = r"^[0-9]+$"
        if allow_empty:
            integer_regex += r"|^$"
        validator = Validator.from_callable(
            lambda answer: re.match(integer_regex, answer),
            error_message="invalid integer",
        )
        return validator

    def _get_user_input(self):
        """
        Display appropriate label and collect user input.

        This private method displays the appropriate prompt label
        for the field using the appropriate display function
        and collect the user input.

        Returns
        -------
        user_input : list or str
            A string for fields that accepts a single input
            or a list of strings for fields that accept multiple
            inputs, e.g., subgroups.
        """
        # if we are dealing with a field that accepts multiple inputs
        if self.count == "multiple":
            # instantiate a blank list to hold the multiple values
            values = []

            # show the name of the field as a heading but do not
            # ask for input yet
            print_formatted_text(HTML(f" <b>{self.label}</b>"))

            # ask the user how many of the multiple inputs they
            # intend to provide; this must be non-zero
            num_entries = prompt(
                "  How many do you want to specify: ",
                validator=self._make_integer_validator(),
            )
            num_entries = int(num_entries)

            # display secondary prompts, one for each of the inputs
            # with the appropriate completer, validator, and style
            for i in range(num_entries):
                value = prompt(
                    f"   Enter #{i+1}: ",
                    completer=self.completer,
                    validator=self.validator,
                    complete_style=self.complete_style,
                )
                # save the value in the list
                values.append(value)

            # this is what we will return
            user_input = values

        # if we are dealing with a simple single-input field
        else:
            # nothing fancy, just display the label, attach
            # the appropriate completer, validator, and style,
            # and get the user input
            user_input = prompt(
                HTML(f" <b>{self.label}</b>: "),
                completer=self.completer,
                validator=self.validator,
                complete_style=self.complete_style,
            )

        return user_input

    def _finalize(self, user_input):
        """
        Convert given input to appropriate type.

        This private method takes the provided user input
        and converts it to the appropriate type.

        Parameters
        ----------
        user_input : list or str
            Description

        Returns
        -------
        final value
            The converted value.
        """
        if (user_input == "" or user_input == []) and self.field_type == "optional":
            final_value = DEFAULTS.get(self.field_name)
        else:
            # boolean fields need to be converted to actual booleans
            if self.data_type == "boolean":
                final_value = False if user_input == "false" else True
            # and integer fields to integers/None
            elif self.data_type == "integer":
                final_value = int(user_input)
            else:
                final_value = user_input

        return final_value

    def get_value(self):
        """
        Get value of instantiated interactive field.

        This is the main public method for this class.

        Returns
        -------
        final_value : list or str
            The final value of the field which may be a list of
            strings or a string.
        """
        # use a while loop to keep asking for the user input
        # until the user either enters it or uses ctrl-D
        # to indicate that they do not want to; ctrl-c
        # just cancels the current entry and asks again
        while True:
            try:
                sys.stderr.write("\n")
                user_input = self._get_user_input()
                final_value = self._finalize(user_input)
            except KeyboardInterrupt:
                continue
            else:
                return final_value


class ConfigurationGenerator:
    """
    Class to encapsulate automated batch-mode and interactive generation.

    Attributes
    ----------
    context : str
        Name of the command-line tool for which we are generating the
        configuration file.
    as_string : bool, optional
        If ``True``, return a formatted and indented string representation
        of the configuration, rather than a dictionary. Note that this only
        affects the batch-mode generation. Interactive generation always
        returns a string.
        Defaults to ``False``.
    suppress_warnings : bool, optional
        If ``True``, do not generate any warnings for batch-mode generation.
        Defaults to ``False``.
    use_subgroups : bool, optional
        If ``True``, include subgroup-related sections in the list of general sections
        in the configuration file.
        Defaults to ``False``.
    """

    def __init__(
        self, context, as_string=False, suppress_warnings=False, use_subgroups=False
    ):  # noqa
        self.context = context
        self.use_subgroups = use_subgroups
        self.suppress_warnings = suppress_warnings
        self.as_string = as_string
        self.logger = logging.getLogger(__name__)

        # we need to save the first required and first optional field we will
        # insert since we will use them as sign posts to insert comments later
        self._required_fields = CHECK_FIELDS[self.context]["required"]
        self._optional_fields = sorted(CHECK_FIELDS[self.context]["optional"])
        self._first_required_field = self._required_fields[0]
        self._first_optional_field = self._optional_fields[0]

    def _convert_to_string(
        self,
        config_object,
        insert_url_comment=True,
        insert_required_comment=True,
        insert_optional_comment=True,
    ):
        configuration = str(config_object)

        # insert the URL comment first, right above the first required field
        if insert_url_comment:
            base_url = "https://rsmtool.readthedocs.io/en/stable"
            doc_slug = CONFIGURATION_DOCUMENTATION_SLUGS[self.context]
            doc_url = f"{base_url}/{doc_slug}"
            configuration = re.sub(
                rf'([ ]+)("{self._first_required_field}": [^,]+,\n)',
                rf"\1// Reference: {doc_url}\n\1\2",
                configuration,
            )

        # insert first comment right above the first required field
        if insert_required_comment:
            configuration = re.sub(
                rf'([ ]+)("{self._first_required_field}": [^,]+,\n)',
                rf'\1// REQUIRED: replace "ENTER_VALUE_HERE" with the appropriate value!\n\1\2',  # noqa
                configuration,
            )

        # insert second comment right above the first optional field
        if insert_optional_comment:
            configuration = re.sub(
                rf'([ ]+)("{self._first_optional_field}": [^,]+,\n)',
                r"\1// OPTIONAL: replace default values below based on your data.\n\1\2",
                configuration,
            )

        return configuration

    def _get_all_general_section_names(self):
        default_general_sections_value = DEFAULTS.get("general_sections", "")
        default_special_sections_value = DEFAULTS.get("special_sections", "")
        default_custom_sections_value = DEFAULTS.get("custom_sections", "")

        # if we are told ot use subgroups then just make up a dummy subgroup
        # value so that the subgroup-based sections will be included in the
        # section list. This value is not actually used in configuration file.
        subgroups_value = ["GROUP"] if self.use_subgroups else DEFAULTS.get("subgroups", "")
        return Reporter().determine_chosen_sections(
            default_general_sections_value,
            default_special_sections_value,
            default_custom_sections_value,
            subgroups_value,
            context=self.context,
        )

    def interact(self, output_file_name=None):
        """
        Automatically generate an example configuration in interactive mode.

        Parameters
        ----------
        output_file_name : str, optional
            The file path where the configuration will eventually be saved.
            Note that this function just uses this name to inform the user.
            The actual saving happens elsewhere.

        Returns
        -------
        str
            The generated configuration as a formatted string.

        Note
        ----
        This method should *only* be used in terminals, and not in
        Jupyter notebooks.

        """
        # clear the screen first
        clear()

        # print the preamble and some instructions
        sys.stderr.write("\n")
        sys.stderr.write("Entering interactive mode:\n")
        sys.stderr.write(" - press ctrl-d to exit without generating a configuration\n")
        sys.stderr.write(" - press tab or start typing when choosing files/directories/models\n")
        sys.stderr.write(" - press enter to accept the default value for a field (underlined)\n")
        sys.stderr.write(" - press ctrl-c to cancel current entry for a field and enter again\n")
        sys.stderr.write(" - you may still need to edit the generated configuration\n")
        sys.stderr.write("\n")

        if not self.use_subgroups:
            sys.stderr.write(
                "IMPORTANT: If you have subgroups and didn't specify the '-g' "
                "option, exit now (ctrl-d) and re-run!\n"
            )
            sys.stderr.write("\n")

        if output_file_name:
            sys.stderr.write(f"Your configuration is being written to '{output_file_name}'.\n\n")

        # instantiate a blank dictionary
        configdict = OrderedDict()

        # iterate over the required fields first, and then the (sorted) optional fields
        # keep track of which field type we are currently dealing with
        for field_type, field_name in chain(
            product(["required"], self._required_fields),
            product(["optional"], self._optional_fields),
        ):
            # skip the subgroups field unless we were told to use subgroups
            if field_name == "subgroups" and not self.use_subgroups:
                configdict["subgroups"] = DEFAULTS.get("subgroups")
                continue

            # if the field is not one that is meant to be filled interactively,
            # then just use its default value; for "general_sections", expand it
            # so that it is easy for the user to remove sections
            if field_name not in INTERACTIVE_MODE_METADATA:
                non_interactive_field_value = DEFAULTS.get(field_name, "")
                if field_name == "general_sections":
                    non_interactive_field_value = self._get_all_general_section_names()
                configdict[field_name] = non_interactive_field_value
            else:
                # instantiate the interactive field first
                try:
                    interactive_field = InteractiveField(
                        field_name, field_type, INTERACTIVE_MODE_METADATA[field_name]
                    )
                    configdict[field_name] = interactive_field.get_value()
                # if the user pressed Ctrl-D, then exit out of interactive mode
                # without generating anything and return an empty string
                except EOFError:
                    sys.stderr.write("\n")
                    sys.stderr.write("You exited interactive mode without a configuration.")
                    sys.stderr.write("\n")
                    return ""
                # otherwise get the field value and save it

        # create a Configuration instance from the dictionary we just generated
        sys.stderr.write("\n")
        config_object = Configuration(configdict, configdir=os.getcwd(), context=self.context)
        # convert the Configuration object to a string - we are using
        # a special wrapper method since we also want to insert comments
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
            configdict[required_field] = "ENTER_VALUE_HERE"

        # insert the optional fields in alphabetical order
        for optional_field in self._optional_fields:
            # to make it easy for users to add/remove sections, we should
            # populate the `general_sections` field with an explicit list
            # instead of the default value which is simply ``['all']``. To
            # do this, we can use the reporter class.
            if optional_field == "general_sections":
                configdict["general_sections"] = self._get_all_general_section_names()
            else:
                configdict[optional_field] = DEFAULTS.get(optional_field, "")

        # create a Configuration object
        config_object = Configuration(configdict, configdir=os.getcwd(), context=self.context)

        # if we were asked for string output, then convert this dictionary to
        # a string that will also insert some useful comments
        if self.as_string:
            configuration = self._convert_to_string(config_object)
        # otherwise we just return the dictionary underlying the Configuration object
        else:
            configuration = config_object._config

        # print out a warning to make it clear that it cannot be used as is
        if not self.suppress_warnings:
            self.logger.warning(
                "Automatically generated configuration files MUST "
                "be edited to add values for required fields and "
                "even for optional ones depending on your data."
            )

        # return either the Configuration object or the string
        return configuration
