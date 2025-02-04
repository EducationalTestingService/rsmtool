#!/usr/bin/env python
"""
Compare two rsmtool/rsmeval experiments.

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

import glob
import logging
import sys
from os.path import abspath, exists, join, normpath
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .configuration_parser import Configuration, configure
from .reader import DataReader
from .reporter import Reporter
from .utils.commandline import ConfigurationGenerator, setup_rsmcmd_parser
from .utils.constants import VALID_PARSER_SUBCOMMANDS
from .utils.logging import LogFormatter
from .utils.wandb import init_wandb_run, log_configuration_to_wandb


def check_experiment_id(experiment_dir: str, experiment_id: str) -> None:
    """
    Check that ``experiment_dir`` contains the outputs for ``experiment_id``.

    Parameters
    ----------
    experiment_dir : str
        path to the directory with the experiment output.
    experiment_id : str
        The ID of the original experiment used to generate the output.

    Raises
    ------
    FileNotFoundError
        If ``experiment_dir`` does not contain any outputs for ``experiment_id``.
    """
    # list all possible output files which start with experiment_id
    outputs = glob.glob(join(experiment_dir, "output", f"{experiment_id}_*.*"))

    # raise an error if none exists
    if len(outputs) == 0:
        raise FileNotFoundError(
            f"The directory {experiment_dir} does not contain "
            f"any outputs of an rsmtool experiment {experiment_id}"
        )


def run_comparison(
    config_file_or_obj_or_dict: Union[str, Configuration, Dict[str, Any], Path], output_dir: str
) -> None:
    """
    Run an rsmcompare experiment using the given configuration.

    Use the given configuration file, object, or dictionary and generate
    the report in the given directory.

    Parameters
    ----------
    config_file_or_obj_or_dict : Union[str, Configuration, Dict[str, Any], Path]
        Path to the experiment configuration file either a a string
        or as a ``pathlib.Path`` object. Users can also pass a
        ``Configuration`` object that is in memory or a Python dictionary
        with keys corresponding to fields in the configuration file. Given a
        configuration file, any relative paths in the configuration file
        will be interpreted relative to the location of the file. Given a
        ``Configuration`` object, relative paths will be interpreted
        relative to the ``configdir`` attribute, that _must_ be set. Given
        a dictionary, the reference path is set to the current directory.
    output_dir : str
        Path to the experiment output directory.

    Raises
    ------
    FileNotFoundError
        If either of the two input directories in ``config_file_or_obj_or_dict``
        do not exist.
    FileNotFoundError
        If the directories do not contain rsmtool outputs at all.
    """
    logger = logging.getLogger(__name__)

    configuration = configure("rsmcompare", config_file_or_obj_or_dict)

    logger.info("Saving configuration file.")
    configuration.save(output_dir)

    # If wandb logging is enabled, and wandb_run is not provided,
    # start a wandb run and log configuration
    wandb_run = init_wandb_run(configuration)
    log_configuration_to_wandb(wandb_run, configuration)

    # get the information about the "old" experiment
    experiment_id_old = configuration["experiment_id_old"]
    experiment_dir_old = DataReader.locate_files(
        configuration["experiment_dir_old"], configuration.configdir
    )[0]
    if not experiment_dir_old:
        raise FileNotFoundError(
            f"The directory {configuration['experiment_dir_old']} " f"does not exist."
        )

    csvdir_old = normpath(join(experiment_dir_old, "output"))
    figdir_old = normpath(join(experiment_dir_old, "figure"))
    if not exists(csvdir_old) or not exists(figdir_old):
        raise FileNotFoundError(
            f"The directory {experiment_dir_old} does not contain "
            f"the output of an rsmtool experiment."
        )

    check_experiment_id(experiment_dir_old, experiment_id_old)

    # get the information about the "new" experiment
    experiment_id_new = configuration["experiment_id_new"]
    experiment_dir_new = DataReader.locate_files(
        configuration["experiment_dir_new"], configuration.configdir
    )[0]
    if not experiment_dir_new:
        raise FileNotFoundError(
            f"The directory {configuration['experiment_dir_new']} " f"does not exist."
        )

    csvdir_new = normpath(join(experiment_dir_new, "output"))
    figdir_new = normpath(join(experiment_dir_new, "figure"))
    if not exists(csvdir_new) or not exists(figdir_new):
        raise FileNotFoundError(
            f"The directory {experiment_dir_new} does not contain the "
            f"output of an rsmtool experiment."
        )

    check_experiment_id(experiment_dir_new, experiment_id_new)

    # are there specific general report sections we want to include?
    general_report_sections = configuration["general_sections"]

    custom_report_section_paths = configuration["custom_sections"]

    # if custom report sections exist, locate sections; otherwise, create empty list
    if custom_report_section_paths:
        logger.info("Locating custom report sections")
        custom_report_sections = Reporter.locate_custom_sections(
            custom_report_section_paths, configuration.configdir
        )
    else:
        custom_report_sections = []

    # get the section order
    section_order = configuration["section_order"]

    # get the subgroups if any
    subgroups = configuration.get("subgroups")

    # Initialize reporter
    reporter = Reporter(logger=logger, wandb_run=wandb_run)

    chosen_notebook_files = reporter.get_ordered_notebook_files(
        general_report_sections,
        custom_report_sections,
        section_order,
        subgroups,
        model_type=None,
        context="rsmcompare",
    )

    # add chosen notebook files to configuration
    configuration["chosen_notebook_files"] = chosen_notebook_files

    # now generate the comparison report
    logger.info("Starting report generation.")
    reporter.create_comparison_report(
        configuration, csvdir_old, figdir_old, csvdir_new, figdir_new, output_dir
    )


def main(argv: Optional[List[str]] = None) -> None:
    """
    Entry point for the ``rsmcompare`` command-line tool.

    Parameters
    ----------
    argv : Optional[List[str]]
        List of arguments to use instead of ``sys.argv``.
        Defaults to ``None``.
    """
    # if no arguments are passed, then use sys.argv
    if argv is None:
        argv = sys.argv[1:]

    # set up the basic logging configuration
    formatter = LogFormatter()

    # we need two handlers, one that prints to stdout
    # for the "run" command and one that prints to stderr
    # from the "generate" command; the latter is necessary
    # because do not want the warning to show up in the
    # generated configuration file
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)

    logging.root.setLevel(logging.INFO)
    logger = logging.getLogger(__name__)

    # set up an argument parser via our helper function
    parser = setup_rsmcmd_parser("rsmcompare", uses_output_directory=True, uses_subgroups=True)

    # if we have no arguments at all then just show the help message
    if len(argv) < 1:
        argv.append("-h")

    # if the first argument is not one of the valid sub-commands
    # or one of the valid optional arguments, then assume that they
    # are arguments for the "run" sub-command. This allows the
    # old style command-line invocations to work without modification.
    if argv[0] not in VALID_PARSER_SUBCOMMANDS + [
        "-h",
        "--help",
        "-V",
        "--version",
    ]:
        args_to_pass = ["run"] + argv
    else:
        args_to_pass = argv
    args = parser.parse_args(args=args_to_pass)

    # call the appropriate function based on which sub-command was run
    if args.subcommand == "run":
        # when running, log to stdout
        logging.root.addHandler(stdout_handler)

        # run the experiment
        logger.info(f"Output directory: {args.output_dir}")
        run_comparison(abspath(args.config_file), abspath(args.output_dir))

    else:
        # when generating, log to stderr
        logging.root.addHandler(stderr_handler)

        # auto-generate an example configuration and print it to STDOUT
        generator = ConfigurationGenerator(
            "rsmcompare",
            as_string=True,
            suppress_warnings=args.quiet,
            use_subgroups=args.subgroups,
        )
        configuration = (
            generator.interact(output_file_name=args.output_file.name if args.output_file else None)
            if args.interactive
            else generator.generate()
        )
        print(configuration, file=args.output_file)


if __name__ == "__main__":
    main()
