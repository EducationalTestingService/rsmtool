"""
The script to create a comparison report for multiple models

:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
"""

#!/usr/bin/env python

import argparse
import glob
import logging
import os
import sys

from os import listdir
from os.path import abspath, dirname, exists, join, normpath

import pandas as pd

from rsmtool.input import (check_main_config,
                           locate_file,
                           read_json_file,
                           locate_custom_sections)

from rsmtool.report import (create_summary_report,
                            get_ordered_notebook_files)

from rsmtool.utils import (scale_coefficients,
                           write_experiment_output,
                           write_feature_json)

from rsmtool.utils import LogFormatter

from rsmtool.version import __version__


def check_experiment_dir(experiment_dir, configpath):
    """
    Check that the supplied experiment directory exists and contains
    the output of the rsmtool experiment.

    Parameters
    experiment_dir : str
        Supplied path to the experiment_dir
    configpath: str
        Path to the directory containing the configuration file

    Returns
    -------
    jsons : list
        A list paths to all configuration json files contained in the output directory

    Raises
    ------
    FileNotFoundError
        If the directory does not exist or does not contain and output
        of an RSMTool experiment
    """

    full_path_experiment_dir = locate_file(experiment_dir, configpath)
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

        # find the json config files  for all experiments stored in this directory
        jsons = glob.glob(join(csvdir,  '*.json'))
        if len(jsons) == 0:
            raise FileNotFoundError("The directory {} does not contain "
                                    "the .json configuration files for rsmtool "
                                    "experiments.".format(full_path_experiment_dir))

    return jsons


def run_summary(config_file, output_dir):
    """
    Run RSMSumm experiment using the given configuration
    file and generate all outputs in the given directory.

    Parameters
    ----------
    config_file : str
        Path to the experiment configuration file.
    output_dir : str
        Path to the experiment output directory.

    Raises
    ------
    ValueError
        If any of the required fields are missing or ill-specified.

    """

    logger = logging.getLogger(__name__)

    # load the information from the config file
    # read in the main config file
    config_obj = read_json_file(config_file)
    config_obj = check_main_config(config_obj, context='rsmsumm')

    # get the directory where the config file lives
    configpath = dirname(config_file)

    # get the summary ID
    summary_id = config_obj['summary_id']

    # get the description
    description = config_obj['description']

    # get the list of the experiment dirs
    experiment_dirs = config_obj['experiment_dirs']

    # check the experiment dirs and assemble the list of csvdir and jsons
    all_experiments = []
    for experiment_dir in experiment_dirs:
        experiments = check_experiment_dir(experiment_dir, configpath)
        all_experiments.extend(experiments)

    # get the subgroups if any
    # Note: at the moment no comparison are reported for subgroups.
    # this option is added to the code to make it easier to add
    # subgroup comparisons in future versions
    subgroups = config_obj.get('subgroups')

    general_report_sections = config_obj['general_sections']

    # get any special sections that the user might have specified
    special_report_sections = config_obj['special_sections']

    # get any custom sections and locate them to make sure
    # that they exist, otherwise raise an exception
    custom_report_section_paths = config_obj['custom_sections']
    if custom_report_section_paths:
        logger.info('Locating custom report sections')
        custom_report_sections = locate_custom_sections(custom_report_section_paths, configpath)
    else:
        custom_report_sections = []

    section_order = config_obj['section_order']

    # check all sections values and order and get the
    # ordered list of notebook files
    chosen_notebook_files = get_ordered_notebook_files(general_report_sections,
                                                       special_report_sections,
                                                       custom_report_sections,
                                                       section_order,
                                                       subgroups,
                                                       model_type=None,
                                                       context='rsmsumm')
    # now generate the comparison report
    logger.info('Starting report generation')
    create_summary_report(summary_id, description,
                          all_experiments,
                          output_dir, subgroups,
                          chosen_notebook_files)


def main():
    # set up the basic logging config
    fmt = LogFormatter()
    hdlr = logging.StreamHandler(sys.stdout)
    hdlr.setFormatter(fmt)
    logging.root.addHandler(hdlr)
    logging.root.setLevel(logging.INFO)

    # get the logger
    logger = logging.getLogger(__name__)

    # set up an argument parser
    parser = argparse.ArgumentParser(prog='rsmsumm')

    parser.add_argument('config_file',
                        help="The JSON config file for this experiment")

    parser.add_argument('output_dir', nargs='?', default=os.getcwd(),
                        help="The output directory where all the files "
                             "for this experiment will be stored")

    parser.add_argument('-V', '--version', action='version',
                        version='%(prog)s {0}'.format(__version__))

    # parse given command line arguments
    args = parser.parse_args()
    logger.info('Output directory: {}'.format(args.output_dir))

    # convert all paths to absolute to make sure
    # all files can be found later
    config_file = abspath(args.config_file)
    output_dir = abspath(args.output_dir)

    # make sure that the given config file exists
    if not exists(config_file):
        raise FileNotFoundError('Main config file {} '
                                'not found.'.format(config_file))

    # run the experiment
    run_summary(config_file, output_dir)


if __name__ == '__main__':
    main()
