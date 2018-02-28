#!/usr/bin/env python

"""
The script to create a summary report for experiment

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:date: 10/25/2017
:organization: ETS
"""


import argparse
import glob
import logging
import os
import sys

from os import listdir
from os.path import (abspath,
                     dirname,
                     exists,
                     join,
                     normpath)

from rsmtool import VERSION_STRING
from rsmtool.configuration_parser import ConfigurationParser, Configuration
from rsmtool.reader import DataReader
from rsmtool.reporter import Reporter

from rsmtool.utils import LogFormatter


def check_experiment_dir(experiment_dir, configpath):
    """
    Check that the supplied experiment directory exists and contains
    the output of the rsmtool experiment.

    Parameters
    ----------
    experiment_dir : str
        Supplied path to the experiment_dir.
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

        return jsons


def run_summary(config_file_or_obj, output_dir):
    """
    Run rsmsummarize experiment using the given configuration
    file and generate all outputs in the given directory.

    Parameters
    ----------
    config_file_or_obj : str or configuration_parser.Configuration
        Path to the experiment configuration file.
        Users can also pass a `Configuration` object that is in memory.
    output_dir : str
        Path to the experiment output directory.

    Raises
    ------
    ValueError
        If any of the required fields are missing or ill-specified.
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

    # Allow users to pass Configuration object to the
    # `config_file_or_obj` argument, rather than read file
    if not isinstance(config_file_or_obj, Configuration):

        # Instantiate configuration parser object
        parser = ConfigurationParser.get_configparser(config_file_or_obj)
        configuration = parser.read_normalize_validate_and_process_config(config_file_or_obj,
                                                                          context='rsmsummarize')

        # get the directory where the configuration file lives
        configpath = dirname(config_file_or_obj)

    else:

        configuration = config_file_or_obj
        if configuration.filepath is not None:
            configpath = dirname(configuration.filepath)
        else:
            configpath = os.getcwd()

    # get the list of the experiment dirs
    experiment_dirs = configuration['experiment_dirs']

    # check the experiment dirs and assemble the list of csvdir and jsons
    all_experiments = []
    for experiment_dir in experiment_dirs:
        experiments = check_experiment_dir(experiment_dir, configpath)
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
                                                                 configpath)
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

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)

    # get the logger
    logger = logging.getLogger(__name__)

    # set up an argument parser
    parser = argparse.ArgumentParser(prog='rsmsummarize')

    parser.add_argument('-f', '--force', dest='force_write',
                        action='store_true', default=False,
                        help="If true, rsmsummarize will not check if the"
                             " output directory already contains the "
                             "output of another rsmsummarize experiment. ")

    parser.add_argument('config_file',
                        help="The JSON configuration file for this experiment")

    parser.add_argument('output_dir', nargs='?', default=os.getcwd(),
                        help="The output directory where all the files "
                             "for this experiment will be stored")

    parser.add_argument('-V', '--version', action='version',
                        version=VERSION_STRING)

    # parse given command line arguments
    args = parser.parse_args()
    logger.info('Output directory: {}'.format(args.output_dir))

    # Raise an error if the specified output directory
    # already contains a non-empty `output` directory, unless
    # `--force` was specified, in which case we assume
    # that the user knows what she is doing and simply
    # output a warning saying that the report might
    # not be correct.
    csvdir = join(args.output_dir, 'output')
    non_empty_csvdir = exists(csvdir) and listdir(csvdir)
    if non_empty_csvdir:
        if not args.force_write:
            raise IOError("'{}' already contains a non-empty 'output' "
                          "directory.".format(args.output_dir))
        else:
            logger.warning("{} already contains a non-empty 'output' directory. "
                           "The generated report might contain "
                           "unexpected information from a previous "
                           "experiment.".format(args.output_dir))

    # convert all paths to absolute to make sure
    # all files can be found later
    config_file = abspath(args.config_file)
    output_dir = abspath(args.output_dir)

    # make sure that the given configuration file exists
    if not exists(config_file):
        raise FileNotFoundError('Main configuration file {} '
                                'not found.'.format(config_file))

    # run the experiment
    run_summary(config_file, output_dir)


if __name__ == '__main__':

    main()
