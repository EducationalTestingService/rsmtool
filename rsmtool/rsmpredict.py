#!/usr/bin/env python

"""
Utility to generate predictions on new data
from existing RSMTool models.

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

import glob
import logging
import os
import sys

from os.path import (abspath,
                     basename,
                     dirname,
                     exists,
                     join,
                     normpath,
                     splitext,
                     split)

from .configuration_parser import configure
from .modeler import Modeler
from .preprocessor import FeaturePreprocessor
from .reader import DataReader
from .utils.commandline import (ConfigurationGenerator,
                                CmdOption,
                                setup_rsmcmd_parser)
from .utils.constants import VALID_PARSER_SUBCOMMANDS
from .utils.logging import LogFormatter
from .writer import DataWriter


def compute_and_save_predictions(config_file_or_obj_or_dict,
                                 output_file,
                                 feats_file=None):
    """
    Run ``rsmpredict`` with given configuration file and generate
    predictions (and, optionally, pre-processed feature values).

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
    output_file : str
        The path to the output file.
    feats_file : str, optional
        Path to the output file for saving preprocessed feature values.

    Raises
    ------
    FileNotFoundError
        If any of the files contained in ``config_file_or_obj_or_dict`` cannot
        be located, or if ``experiment_dir`` does not exist, or if ``experiment_dir``
        does not contain the required output needed from an rsmtool experiment.
    RuntimeError
        If the name of the output file does not end in '.csv', '.tsv', or '.xlsx'.
    """

    logger = logging.getLogger(__name__)

    configuration = configure('rsmpredict', config_file_or_obj_or_dict)

    # get the experiment ID
    experiment_id = configuration['experiment_id']

    # Get output format
    file_format = configuration.get('file_format', 'csv')

    # Get DataWriter object
    writer = DataWriter(experiment_id)

    # get the input file containing the feature values
    # for which we want to generate the predictions
    input_features_file = DataReader.locate_files(configuration['input_features_file'],
                                                  configuration.configdir)
    if not input_features_file:
        raise FileNotFoundError('Input file {} does not exist'
                                ''.format(configuration['input_features_file']))

    experiment_dir = DataReader.locate_files(configuration['experiment_dir'],
                                             configuration.configdir)
    if not experiment_dir:
        raise FileNotFoundError('The directory {} does not exist.'
                                ''.format(configuration['experiment_dir']))
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

    logger.info('Reading input files.')

    feature_info = join(experiment_output_dir,
                        '{}_feature.csv'.format(experiment_id))

    post_processing = join(experiment_output_dir,
                           '{}_postprocessing_params.csv'.format(experiment_id))

    file_paths = [input_features_file, feature_info, post_processing]
    file_names = ['input_features',
                  'feature_info',
                  'postprocessing_params']

    converters = {'input_features': configuration.get_default_converter()}

    # Initialize the reader
    reader = DataReader(file_paths, file_names, converters)
    data_container = reader.read(kwargs_dict={'feature_info': {'index_col': 0}})

    # load the Modeler to generate the predictions
    model = Modeler.load_from_file(join(experiment_output_dir,
                                        '{}.model'.format(experiment_id)))

    # Add the model to the configuration object
    configuration['model'] = model

    # Initialize the processor
    processor = FeaturePreprocessor()

    (processed_config,
     processed_container) = processor.process_data(configuration,
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

    # to set up the argument parser, we first need to instantiate options
    # specific to rsmpredictso we use the `CmdOption` namedtuples
    non_standard_options = [CmdOption(dest='output_file',
                                      help="output file where predictions will be saved."),
                            CmdOption(dest='preproc_feats_file',
                                      help="if specified, the preprocessed features "
                                           "will be saved in this file",
                                      longname='features',
                                      required=False)]

    # now call the helper function to instantiate the parser for us
    parser = setup_rsmcmd_parser('rsmpredict',
                                 uses_output_directory=False,
                                 extra_run_options=non_standard_options)

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
        preproc_feats_file = None
        if args.preproc_feats_file:
            preproc_feats_file = abspath(args.preproc_feats_file)
        compute_and_save_predictions(abspath(args.config_file),
                                     abspath(args.output_file),
                                     feats_file=preproc_feats_file)

    else:

        # when generating, log to stderr
        logging.root.addHandler(stderr_handler)

        # auto-generate an example configuration and print it to STDOUT
        generator = ConfigurationGenerator('rsmpredict',
                                           as_string=True,
                                           suppress_warnings=args.quiet)
        configuration = generator.interact() if args.interactive else generator.generate()
        print(configuration)


if __name__ == '__main__':
    main()
