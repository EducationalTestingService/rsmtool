#!/usr/bin/env python

"""
Utility to generate predictions on new data
from existing RSMTool models.

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

from os.path import (basename,
                     dirname,
                     exists,
                     join,
                     normpath,
                     splitext,
                     split)

from rsmtool import VERSION_STRING
from rsmtool.configuration_parser import ConfigurationParser, Configuration
from rsmtool.modeler import Modeler
from rsmtool.preprocessor import FeaturePreprocessor
from rsmtool.reader import DataReader
from rsmtool.utils import LogFormatter
from rsmtool.writer import DataWriter


def compute_and_save_predictions(config_file_or_obj, output_file, feats_file=None):
    """
    Run ``rsmpredict`` with given configuration file and generate
    predictions (and, optionally, pre-processed feature values).

    Parameters
    ----------
    config_file_or_obj : str or configuration_parser.Configuration
        Path to the experiment configuration file.
        Users can also pass a `Configuration` object that is in memory.
    output_dir : str
        Path to the output directory for saving files.
    feats_file (optional): str
        Path to the output file for saving preprocessed feature values.

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
        config = parser.read_normalize_validate_and_process_config(config_file_or_obj,
                                                                   context='rsmpredict')

        # get the directory where the config file lives
        configpath = dirname(config_file_or_obj)

    else:

        config = config_file_or_obj
        if config.filepath is not None:
            configpath = dirname(config.filepath)
        else:
            configpath = os.getcwd()

    # get the experiment ID
    experiment_id = config['experiment_id']

    # Get output format
    file_format = config.get('file_format', 'csv')

    # Get DataWriter object
    writer = DataWriter(experiment_id)

    # get the input file containing the feature values
    # for which we want to generate the predictions
    input_features_file = DataReader.locate_files(config['input_features_file'], configpath)
    if not input_features_file:
        raise FileNotFoundError('Input file {} does not exist'
                                ''.format(config['input_features_file']))

    experiment_dir = DataReader.locate_files(config['experiment_dir'], configpath)
    if not experiment_dir:
        raise FileNotFoundError('The directory {} does not exist.'
                                ''.format(config['experiment_dir']))
    else:
        experiment_output_dir = normpath(join(experiment_dir, 'output'))
        if not exists(experiment_output_dir):
            raise FileNotFoundError('The directory {} does not contain '
                                    'the output of an rsmtool experiment.'.format(experiment_dir))

    # find all the .model files in the experiment output directory
    model_files = glob.glob(join(experiment_output_dir, '*.model'))
    if not model_files:
        raise FileNotFoundError('The directory {} does not contain any rsmtool models.'
                                ''.format(experiment_output_dir))

    experiment_ids = [splitext(basename(mf))[0] for mf in model_files]
    if experiment_id not in experiment_ids:
        raise FileNotFoundError('{} does not contain a model for the experiment "{}". '
                                'The following experiments are contained in this '
                                'directory: {}'.format(experiment_output_dir,
                                                       experiment_id,
                                                       experiment_ids))

    # check that the directory contains outher required files
    required_file_types = ['feature', 'postprocessing_params']
    for file_type in required_file_types:
        expected_file_name = "{}_{}.csv".format(experiment_id, file_type)
        if not exists(join(experiment_output_dir, expected_file_name)):
            raise FileNotFoundError('{} does not contain the required file '
                                    '{} that was generated during the '
                                    'original model training'.format(experiment_output_dir,
                                                                     expected_file_name))

    # model_files = glob.glob(join(experiment_output_dir, '*.model'))
    # if not model_files:
    #     raise FileNotFoundError('The directory {} does not contain any rsmtool models. '
    #                             ''.format(experiment_output_dir))

    logger.info('Reading input files.')

    feature_info = join(experiment_output_dir,
                        '{}_feature.csv'.format(experiment_id))

    post_processing = join(experiment_output_dir,
                           '{}_postprocessing_params.csv'.format(experiment_id))

    file_paths = [input_features_file, feature_info, post_processing]
    file_names = ['input_features',
                  'feature_info',
                  'postprocessing_params']

    converters = {'input_features': config.get_default_converter()}

    # Initialize the reader
    reader = DataReader(file_paths, file_names, converters)
    data_container = reader.read(kwargs_dict={'feature_info': {'index_col': 0}})

    # load the Modeler to generate the predictions
    model = Modeler.load_from_file(join(experiment_output_dir,
                                        '{}.model'.format(experiment_id)))

    # Add the model to the configuration object
    config['model'] = model

    # Initialize the processor
    processor = FeaturePreprocessor()

    (processed_config,
     processed_container) = processor.process_data(config,
                                                   data_container,
                                                   context='rsmpredict')

    # save the pre-processed features to disk if we were asked to
    if feats_file is not None:
        logger.info('Saving pre-processed feature values to {}'.format(feats_file))

        feats_dir = dirname(feats_file)

        # create any directories needed for the output file
        os.makedirs(feats_dir, exist_ok=True)

        _, feats_filename = split(feats_file)
        feats_filename, _ = splitext(feats_filename)

        # Write out files
        writer.write_experiment_output(feats_dir,
                                       processed_container,
                                       include_experiment_id=False,
                                       dataframe_names=['features_processed'],
                                       new_names_dict={'features_processed':
                                                       feats_filename},
                                       file_format=file_format)

    if (output_file.lower().endswith('.csv') or
            output_file.lower().endswith('.xlsx')):

        output_dir = dirname(output_file)
        _, filename = split(output_file)
        filename, _ = splitext(filename)

    else:
        output_dir = output_file
        filename = 'predictions_with_metadata'

    # create any directories needed for the output file
    os.makedirs(output_dir, exist_ok=True)

    # save the predictions to disk
    logger.info('Saving predictions.')

    # Write out files
    writer.write_experiment_output(output_dir,
                                   processed_container,
                                   include_experiment_id=False,
                                   dataframe_names=['predictions_with_metadata'],
                                   new_names_dict={'predictions_with_metadata':
                                                   filename},
                                   file_format=file_format)

    # save excluded responses to disk
    if not processed_container.excluded.empty:

        # save the predictions to disk
        logger.info('Saving excluded responses to {}'.format(join(output_dir,
                                                                  '{}_excluded_responses.csv'
                                                                  ''.format(filename))))

        # Write out files
        writer.write_experiment_output(output_dir,
                                       processed_container,
                                       include_experiment_id=False,
                                       dataframe_names=['excluded'],
                                       new_names_dict={'excluded':
                                                       '{}_excluded_responses'
                                                       ''.format(filename)},
                                       file_format=file_format)


def main():

    # set up the basic logging config
    formatter = LogFormatter()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)

    # set up an argument parser
    parser = argparse.ArgumentParser(prog='rsmpredict')

    parser.add_argument('config_file', help="The JSON config file "
                                            "needed to run rsmpredict")

    parser.add_argument('output_file', help="Output file where "
                                            "predictions will be saved")

    parser.add_argument('--features', dest='preproc_feats_file',
                        help="Output file to save the pre-processed "
                             "version of the features",
                        required=False,
                        default=None)

    parser.add_argument('-V', '--version', action='version',
                        version=VERSION_STRING)

    # parse given command line arguments
    args = parser.parse_args()

    # convert all paths to absolute to make sure
    # all files can be found later
    config_file = os.path.abspath(args.config_file)
    output_file = os.path.abspath(args.output_file)
    preproc_feats_file = None
    if args.preproc_feats_file:
        preproc_feats_file = os.path.abspath(args.preproc_feats_file)

    # generate and save the predictions
    compute_and_save_predictions(config_file,
                                 output_file,
                                 preproc_feats_file)


if __name__ == '__main__':

    main()
