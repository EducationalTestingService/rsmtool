#!/usr/bin/env python

"""
Script to compare two RSMTool experiments.

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

import glob
import logging
import sys

from os.path import abspath, exists, join, normpath

from .configuration_parser import configure
from .reader import DataReader
from .reporter import Reporter
from .utils.commandline import ConfigurationGenerator, setup_rsmcmd_parser
from .utils.constants import VALID_PARSER_SUBCOMMANDS
from .utils.logging import LogFormatter


def check_experiment_id(experiment_dir, experiment_id):
    """
    Check that the supplied ``experiment_dir`` contains
    the outputs for the supplied ``experiment_id``.

    Parameters
    ----------
    experiment_dir : str
        path to the directory with the experiment output
    experiment_id : str
        experiment_id of the original experiment used to generate the
        output

    Raises
    ------
    FileNotFoundError
        if the ``experument_dir`` does not contain any outputs
        for the ``experiment_id``
    """

    # list all possible output files which start with
    # experiment_id
    outputs = glob.glob(join(experiment_dir,
                             'output',
                             '{}_*.*'.format(experiment_id)))

    # raise an error if none exists
    if len(outputs) == 0:
        raise FileNotFoundError("The directory {} does not contain "
                                "any outputs of an rsmtool experiment "
                                "{}".format(experiment_dir, experiment_id))


def run_comparison(config_file_or_obj_or_dict, output_dir):
    """
    Run an ``rsmcompare`` experiment using the given configuration
    file and generate the report in the given directory.

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

    Raises
    ------
    FileNotFoundError
        If either of the two input directories in ``config_file_or_obj_or_dict``
        do not exist, or if the directories do not contain rsmtool outputs at all.
    """

    logger = logging.getLogger(__name__)

    configuration = configure('rsmcompare', config_file_or_obj_or_dict)

    logger.info('Saving configuration file.')
    configuration.save(output_dir)

    # get the information about the "old" experiment
    experiment_id_old = configuration['experiment_id_old']
    experiment_dir_old = DataReader.locate_files(configuration['experiment_dir_old'],
                                                 configuration.configdir)
    if not experiment_dir_old:
        raise FileNotFoundError("The directory {} "
                                "does not exist.".format(configuration['experiment_dir_old']))

    csvdir_old = normpath(join(experiment_dir_old, 'output'))
    figdir_old = normpath(join(experiment_dir_old, 'figure'))
    if not exists(csvdir_old) or not exists(figdir_old):
        raise FileNotFoundError("The directory {} does not contain "
                                "the output of an rsmtool "
                                "experiment.".format(experiment_dir_old))

    check_experiment_id(experiment_dir_old, experiment_id_old)

    # get the information about the "new" experiment
    experiment_id_new = configuration['experiment_id_new']
    experiment_dir_new = DataReader.locate_files(configuration['experiment_dir_new'],
                                                 configuration.configdir)
    if not experiment_dir_new:
        raise FileNotFoundError("The directory {} "
                                "does not exist.".format(configuration['experiment_dir_new']))

    csvdir_new = normpath(join(experiment_dir_new, 'output'))
    figdir_new = normpath(join(experiment_dir_new, 'figure'))
    if not exists(csvdir_new) or not exists(figdir_new):
        raise FileNotFoundError("The directory {} does not contain "
                                "the output of an rsmtool "
                                "experiment.".format(experiment_dir_new))

    check_experiment_id(experiment_dir_new, experiment_id_new)

    # are there specific general report sections we want to include?
    general_report_sections = configuration['general_sections']

    # what about the special or custom sections?
    special_report_sections = configuration['special_sections']

    custom_report_section_paths = configuration['custom_sections']

    # if custom report sections exist, locate sections; otherwise, create empty list
    if custom_report_section_paths:
        logger.info('Locating custom report sections')
        custom_report_sections = Reporter.locate_custom_sections(custom_report_section_paths,
                                                                 configuration.configdir)
    else:
        custom_report_sections = []

    # get the section order
    section_order = configuration['section_order']

    # get the subgroups if any
    subgroups = configuration.get('subgroups')

    # Initialize reporter
    reporter = Reporter()

    chosen_notebook_files = reporter.get_ordered_notebook_files(general_report_sections,
                                                                special_report_sections,
                                                                custom_report_sections,
                                                                section_order,
                                                                subgroups,
                                                                model_type=None,
                                                                context='rsmcompare')

    # add chosen notebook files to configuration
    configuration['chosen_notebook_files'] = chosen_notebook_files

    # now generate the comparison report
    logger.info('Starting report generation.')
    reporter.create_comparison_report(configuration,
                                      csvdir_old,
                                      figdir_old,
                                      csvdir_new,
                                      figdir_new,
                                      output_dir)


def main():

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
    parser = setup_rsmcmd_parser('rsmcompare',
                                 uses_output_directory=True,
                                 uses_subgroups=True)

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
        run_comparison(abspath(args.config_file),
                       abspath(args.output_dir))

    else:

        # when generating, log to stderr
        logging.root.addHandler(stderr_handler)

        # auto-generate an example configuration and print it to STDOUT
        generator = ConfigurationGenerator('rsmcompare',
                                           as_string=True,
                                           suppress_warnings=args.quiet,
                                           use_subgroups=args.subgroups)
        configuration = generator.interact() if args.interactive else generator.generate()
        print(configuration)


if __name__ == '__main__':

    main()
