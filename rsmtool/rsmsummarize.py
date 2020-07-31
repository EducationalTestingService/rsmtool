#!/usr/bin/env python

"""
The script to create a summary report for experiment

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

import glob
import logging
import os
import sys

from os import listdir
from os.path import (abspath,
                     exists,
                     join,
                     normpath)

from .configuration_parser import configure
from .reader import DataReader
from .reporter import Reporter
from .utils.commandline import ConfigurationGenerator, setup_rsmcmd_parser
from .utils.constants import VALID_PARSER_SUBCOMMANDS
from .utils.logging import LogFormatter


def check_experiment_dir(experiment_dir,
                         experiment_name,
                         configpath):
    """
    Check that the supplied experiment directory exists and contains
    the output of the rsmtool experiment.

    Parameters
    ----------
    experiment_dir : str
        Supplied path to the experiment directory.
    experiment_name : str
        The name of the rsmtool experiment we are interested in
    configpath : str
        Path to the directory containing the configuration file.

    Returns
    -------
    jsons : list
        A list paths to all configuration json files contained in the output directory

    Raises
    ------
    FileNotFoundError
        If the directory does not exist or does not contain and output
        of an RSMTool experiment.
    ValueError
        If the given experiment directory contains several JSON configuration
        files instead of just one.
    """
    full_path_experiment_dir = DataReader.locate_files(experiment_dir, configpath)
    if not full_path_experiment_dir:
        raise FileNotFoundError("The directory {} "
                                "does not exist.".format(experiment_dir))
    else:
        # check that there is an output directory
        csvdir = normpath(join(full_path_experiment_dir, 'output'))
        if not exists(csvdir):
            raise FileNotFoundError("The directory {} does not contain "
                                    "the output of an rsmtool "
                                    "experiment.".format(full_path_experiment_dir))

        # find the json configuration files for all experiments stored in this directory
        jsons = glob.glob(join(csvdir, '*.json'))
        if len(jsons) == 0:
            raise FileNotFoundError("The directory {} does not contain "
                                    "the .json configuration files for rsmtool "
                                    "experiments.".format(full_path_experiment_dir))

        # Raise an error if the user specified a list of experiment names
        # but we found several .jsons in the same directory
        if experiment_name and len(jsons) > 1:
            raise ValueError("{} seems to contain the output of multiple experiments. "
                             "In order to use custom experiment names, you must have "
                             "a separate directory "
                             "for each experiment".format(full_path_experiment_dir))

        # return [(json, experiment_name)] when we have experiment name or
        # [(json, None)] if no experiment name has been specified.
        # If the folder contains the output of multiple experiments, return
        # [(json1, None), (json2, None) .... ]
        return list(zip(jsons, [experiment_name] * len(jsons)))


def run_summary(config_file_or_obj_or_dict,
                output_dir,
                overwrite_output=False):
    """
    Run rsmsummarize experiment using the given configuration
    file and generate all outputs in the given directory.

    If ``overwrite_output`` is ``True``, overwrite any existing
    output in the given ``output_dir``.

    Parameters
    ----------
    config_file_or_obj_or_dict : str or pathlib.Path or dict or Configuration
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
    overwrite_output : bool, optional
        If ``True``, overwrite any existing output under ``output_dir``.
        Defaults to ``False``.

    Raises
    ------
    IOError
        If ``output_dir`` already contains the output of a previous experiment
        and ``overwrite_output`` is ``False``.
    """
    logger = logging.getLogger(__name__)

    # create the 'output' and the 'figure' sub-directories
    # where all the experiment output such as the CSV files
    # and the box plots will be saved
    csvdir = abspath(join(output_dir, 'output'))
    figdir = abspath(join(output_dir, 'figure'))
    reportdir = abspath(join(output_dir, 'report'))

    os.makedirs(csvdir, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(reportdir, exist_ok=True)

    # Raise an error if the specified output directory
    # already contains a non-empty `output` directory, unless
    # `overwrite_output` was specified, in which case we assume
    # that the user knows what she is doing and simply
    # output a warning saying that the report might
    # not be correct.
    non_empty_csvdir = exists(csvdir) and listdir(csvdir)
    if non_empty_csvdir:
        if not overwrite_output:
            raise IOError("'{}' already contains a non-empty 'output' "
                          "directory.".format(output_dir))
        else:
            logger.warning("{} already contains a non-empty 'output' directory. "
                           "The generated report might contain "
                           "unexpected information from a previous "
                           "experiment.".format(output_dir))

    configuration = configure('rsmsummarize', config_file_or_obj_or_dict)

    logger.info('Saving configuration file.')
    configuration.save(output_dir)

    # get the list of the experiment dirs
    experiment_dirs = configuration['experiment_dirs']

    # Get experiment names if any
    experiment_names = configuration.get('experiment_names')
    experiment_names = experiment_names if experiment_names else [None] * len(experiment_dirs)
    dirs_with_names = zip(experiment_dirs, experiment_names)

    # check the experiment dirs and assemble the list of csvdir and jsons
    all_experiments = []
    for (experiment_dir, experiment_name) in dirs_with_names:
        experiments = check_experiment_dir(experiment_dir,
                                           experiment_name,
                                           configuration.configdir)
        all_experiments.extend(experiments)

    # get the subgroups if any
    # Note: at the moment no comparison are reported for subgroups.
    # this option is added to the code to make it easier to add
    # subgroup comparisons in future versions
    subgroups = configuration.get('subgroups')

    general_report_sections = configuration['general_sections']

    # get any special sections that the user might have specified
    special_report_sections = configuration['special_sections']

    # get any custom sections and locate them to make sure
    # that they exist, otherwise raise an exception
    custom_report_section_paths = configuration['custom_sections']
    if custom_report_section_paths:
        logger.info('Locating custom report sections')
        custom_report_sections = Reporter.locate_custom_sections(custom_report_section_paths,
                                                                 configuration.configdir)
    else:
        custom_report_sections = []

    section_order = configuration['section_order']

    # Initialize reporter
    reporter = Reporter()

    # check all sections values and order and get the
    # ordered list of notebook files
    chosen_notebook_files = reporter.get_ordered_notebook_files(general_report_sections,
                                                                special_report_sections,
                                                                custom_report_sections,
                                                                section_order,
                                                                subgroups,
                                                                model_type=None,
                                                                context='rsmsummarize')

    # add chosen notebook files to configuration
    configuration['chosen_notebook_files'] = chosen_notebook_files

    # now generate the comparison report
    logger.info('Starting report generation')
    reporter.create_summary_report(configuration,
                                   all_experiments,
                                   csvdir)


def main():

    # set up the basic logging configuration
    formatter = LogFormatter()

    # we need two handlers, one that prints to stdout
    # for the "run" command and one that prints to stderr
    # from the "generate" command; the latter is important
    # because do not want the warning to show up in the
    # generated configuration file
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)

    logging.root.setLevel(logging.INFO)
    logger = logging.getLogger(__name__)

    # set up an argument parser via our helper function
    parser = setup_rsmcmd_parser('rsmsummarize',
                                 uses_output_directory=True,
                                 allows_overwriting=True)

    # if we have no arguments at all then just show the help message
    if len(sys.argv) < 2:
        sys.argv.append("-h")

    # if the first argument is not one of the valid sub-commands
    # or one of the valid optional arguments, then assume that they
    # are arguments for the "run" sub-command. This allows the
    # old style command-line invocations to work without modification.
    if sys.argv[1] not in VALID_PARSER_SUBCOMMANDS + ['-h', '--help',
                                                      '-V', '--version']:
        args_to_pass = ['run'] + sys.argv[1:]
    else:
        args_to_pass = sys.argv[1:]
    args = parser.parse_args(args=args_to_pass)

    # call the appropriate function based on which sub-command was run
    if args.subcommand == 'run':

        # when running, log to stdout
        logging.root.addHandler(stdout_handler)

        # run the experiment
        logger.info('Output directory: {}'.format(args.output_dir))
        run_summary(abspath(args.config_file),
                    abspath(args.output_dir),
                    overwrite_output=args.force_write)

    else:

        # when generating, log to stderr
        logging.root.addHandler(stderr_handler)

        # auto-generate an example configuration and print it to STDOUT
        generator = ConfigurationGenerator('rsmsummarize',
                                           as_string=True,
                                           suppress_warnings=args.quiet)
        configuration = generator.interact() if args.interactive else generator.generate()
        print(configuration)


if __name__ == '__main__':
    main()
