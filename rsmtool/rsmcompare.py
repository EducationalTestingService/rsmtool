#!/usr/bin/env python

"""
Script to compare two RSMTool experiments.

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


def run_comparison(config_file_or_obj, output_dir):
    """
    Run an ``rsmcompare`` experiment using the given configuration
    file and generate the report in the given directory.

    Parameters
    ----------
    config_file_or_obj : str or Configuration
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

    # Allow users to pass Configuration object to the
    # `config_file_or_obj` argument, rather than read file
    if not isinstance(config_file_or_obj, Configuration):

        # Instantiate configuration parser object
        parser = ConfigurationParser.get_configparser(config_file_or_obj)
        configuration = parser.read_normalize_validate_and_process_config(config_file_or_obj,
                                                                          context='rsmcompare')

        # get the directory where the configuration file lives
        configpath = dirname(config_file_or_obj)

    else:
        configuration = config_file_or_obj
        if configuration.filepath is not None:
            configpath = dirname(configuration.filepath)
        else:
            configpath = os.getcwd()

    # get the information about the "old" experiment
    experiment_id_old = configuration['experiment_id_old']
    experiment_dir_old = DataReader.locate_files(configuration['experiment_dir_old'], configpath)
    if not experiment_dir_old:
        raise FileNotFoundError("The directory {} "
                                "does not exist.".format(configuration['experiment_dir_old']))
    else:
        csvdir_old = normpath(join(experiment_dir_old, 'output'))
        figdir_old = normpath(join(experiment_dir_old, 'figure'))
        if not exists(csvdir_old) or not exists(figdir_old):
            raise FileNotFoundError("The directory {} does not contain "
                                    "the output of an rsmtool "
                                    "experiment.".format(experiment_dir_old))

    check_experiment_id(experiment_dir_old, experiment_id_old)

    # get the information about the "new" experiment
    experiment_id_new = configuration['experiment_id_new']
    experiment_dir_new = DataReader.locate_files(configuration['experiment_dir_new'], configpath)
    if not experiment_dir_new:
        raise FileNotFoundError("The directory {} "
                                "does not exist.".format(configuration['experiment_dir_new']))
    else:
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
                                                                 configpath)
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

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)

    # get a logger
    logger = logging.getLogger(__name__)

    # set up an argument parser
    parser = argparse.ArgumentParser(prog='rsmcompare')

    parser.add_argument('config_file', help="The JSON configuration file for "
                                            "this comparison")

    parser.add_argument('output_dir', nargs='?', default=os.getcwd(),
                        help="The output directory where the report "
                             "files for this comparison will be stored")

    parser.add_argument('-V', '--version', action='version',
                        version=VERSION_STRING)

    # parse given command line arguments
    args = parser.parse_args()
    logger.info('Output directory: {}'.format(args.output_dir))

    # convert all paths to absolute to make sure
    # all files can be found later
    config_file = abspath(args.config_file)
    output_dir = abspath(args.output_dir)

    # make sure that the given configuration file exists
    if not exists(config_file):
        raise FileNotFoundError("Main configuration file {} not "
                                "found.".format(config_file))

    # generate a comparison report
    run_comparison(config_file, output_dir)


if __name__ == '__main__':

    main()
