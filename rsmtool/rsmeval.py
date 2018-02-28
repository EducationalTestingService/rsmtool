#!/usr/bin/env python

"""
Run evaluation only experiments.

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:date: 10/25/2017
:organization: ETS
"""


import argparse
import logging
import os
import sys

from os import listdir
from os.path import abspath, exists, join, dirname

from rsmtool import VERSION_STRING
from rsmtool.analyzer import Analyzer
from rsmtool.configuration_parser import ConfigurationParser, Configuration
from rsmtool.preprocessor import FeaturePreprocessor
from rsmtool.reader import DataReader
from rsmtool.reporter import Reporter
from rsmtool.utils import LogFormatter
from rsmtool.writer import DataWriter


def run_evaluation(config_file_or_obj, output_dir):
    """
    Run an `rsmeval` experiment using the given configuration
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
                                                                          context='rsmeval')

        # get the directory where the configuration file lives
        configpath = dirname(config_file_or_obj)

    else:

        configuration = config_file_or_obj
        if configuration.filepath is not None:
            configpath = dirname(configuration.filepath)
        else:
            configpath = os.getcwd()

    logger.info('Saving configuration file.')
    configuration.save(output_dir)

    # Get output format
    file_format = configuration.get('file_format', 'csv')

    # Get DataWriter object
    writer = DataWriter(configuration['experiment_id'])

    # Make sure prediction file can be located
    if not DataReader.locate_files(configuration['predictions_file'],
                                   configpath):
        raise FileNotFoundError('Error: Predictions file {} '
                                'not found.\n'.format(configuration['predictions_file']))

    scale_with = configuration.get('scale_with')

    # scale_with can be one of the following:
    # (a) None       : the predictions are assumed to be 'raw' and should be used as is
    #                  when computing the metrics; the names for the final columns are
    #                  'raw', 'raw_trim' and 'raw_trim_round'.
    # (b) 'asis'     : the predictions are assumed to be pre-scaled and should be used as is
    #                  when computing the metrics; the names for the final columns are
    #                  'scale', 'scale_trim' and 'scale_trim_round'.
    # (c) a CSV file : the predictions are assumed to be 'raw' and should be scaled
    #                  before computing the metrics; the names for the final columns are
    #                  'scale', 'scale_trim' and 'scale_trim_round'.

    # Check whether we want to do scaling
    do_scaling = (scale_with is not None and
                  scale_with != 'asis')

    # The paths to files and names for data container properties
    paths = ['predictions_file']
    names = ['predictions']

    # If we want to do scaling, get the scale file
    if do_scaling:

        # Make sure scale file can be located
        scale_file_location = DataReader.locate_files(scale_with,
                                                      configpath)
        if not scale_file_location:
            raise FileNotFoundError('Could not find scaling file {}.'
                                    ''.format(scale_file_location))

        paths.append('scale_with')
        names.append('scale')

    # Get the paths, names, and converters for the DataReader
    (file_names,
     file_paths) = configuration.get_names_and_paths(paths, names)

    file_paths = DataReader.locate_files(file_paths, configpath)

    converters = {'predictions': configuration.get_default_converter()}

    logger.info('Reading predictions: {}.'.format(configuration['predictions_file']))

    # Initialize the reader
    reader = DataReader(file_paths, file_names, converters)
    data_container = reader.read()

    logger.info('Preprocessing predictions.')

    # Initialize the processor
    processor = FeaturePreprocessor()

    (processed_config,
     processed_container) = processor.process_data(configuration,
                                                   data_container,
                                                   context='rsmeval')

    logger.info('Saving pre-processed predictions and metadata to disk.')
    writer.write_experiment_output(csvdir,
                                   processed_container,
                                   new_names_dict={'pred_test':
                                                   'pred_processed',
                                                   'test_excluded':
                                                   'test_excluded_responses'},
                                   file_format=file_format)

    # Initialize the analyzer
    analyzer = Analyzer()

    # do the data composition stats
    (analyzed_config,
     analyzed_container) = analyzer.run_data_composition_analyses_for_rsmeval(processed_container,
                                                                              processed_config)
    # Write out files
    writer.write_experiment_output(csvdir,
                                   analyzed_container,
                                   file_format=file_format)

    for_pred_data_container = analyzed_container + processed_container

    # run the analyses on the predictions of the model`
    logger.info('Running analyses on predictions.')
    (pred_analysis_config,
     pred_analysis_data_container) = analyzer.run_prediction_analyses(for_pred_data_container,
                                                                      analyzed_config)

    writer.write_experiment_output(csvdir,
                                   pred_analysis_data_container,
                                   reset_index=True,
                                   file_format=file_format)

    # Initialize reporter
    reporter = Reporter()

    # generate the report
    logger.info('Starting report generation.')
    reporter.create_report(processed_config,
                           csvdir,
                           figdir,
                           context='rsmeval')


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
    parser = argparse.ArgumentParser(prog='rsmeval')

    parser.add_argument('-f', '--force', dest='force_write',
                        action='store_true', default=False,
                        help="If true, rsmtool will not check if the"
                             " output directory already contains the "
                             "output of another rsmtool experiment. ")

    parser.add_argument('config_file', help="The JSON configuration file for "
                                            "this experiment")

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
    config_file = os.path.abspath(args.config_file)
    output_dir = os.path.abspath(args.output_dir)

    # make sure that the given configuration file exists
    if not exists(config_file):
        raise FileNotFoundError("Main configuration file {} not "
                                "found.".format(config_file))

    # run the evaluation experiment
    run_evaluation(config_file, output_dir)


if __name__ == '__main__':
    main()
