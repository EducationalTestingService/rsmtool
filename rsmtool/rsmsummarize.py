#!/usr/bin/env python

"""
The script to create a summary report for experiment

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

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


def check_experiment_dir(experiment_dir,
                         experiment_name,
                         configpath):
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
                output_dir):
    """
    Run rsmsummarize experiment using the given configuration
    file and generate all outputs in the given directory.

    Parameters
    ----------
    config_file_or_obj_or_dict : str or Configuration or dict
        Path to the experiment configuration file.
        Users can also pass a `Configuration` object that is in memory
        or a Python dictionary with keys corresponding to fields in the
        configuration file.
        Relative paths in the configuration file will be interpreted relative
        to the location of the file. For configuration object
        `.configdir` needs to be set to indicate the reference path. If
        the user passes a dictionary, the reference path will be set
        to the current directory and all relative paths will be resolved
        relative to this path.
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

    # check what sort of input we got
    # if we got a string we consider this to be path to config file
    if isinstance(config_file_or_obj_or_dict, str):

        # Instantiate configuration parser object
        parser = ConfigurationParser.get_configparser(config_file_or_obj_or_dict)
        configuration = parser.read_normalize_validate_and_process_config(config_file_or_obj_or_dict,
                                                                          context='rsmsummarize')

    elif isinstance(config_file_or_obj_or_dict, dict):

        # initialize the parser from dict
        parser = ConfigurationParser()
        configuration = parser.load_normalize_and_validate_config_from_dict(config_file_or_obj_or_dict,
                                                                            context='rsmsummarize')

    elif isinstance(config_file_or_obj_or_dict, Configuration):

        configuration = config_file_or_obj_or_dict
        # raise an error if we are passed a Configuration object
        # without a configdir attribute. This can only
        # happen if the object was constructed using an earlier version
        # of RSMTool and stored
        if configuration.configdir is None:
            raise AttributeError("Configuration object must have configdir attribute.")

    else:
        raise ValueError("The input to run_summary must be "
                         "a path to the file (str), a dictionary, "
                         "or a configuration object. You passed "
                         "{}.".format(type(config_file_or_obj_or_dict)))
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
