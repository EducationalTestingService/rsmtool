#!/usr/bin/env python

"""
The main RSMTool script.

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:date: 10/25/2017
:organization: ETS
"""

import argparse
import logging
import sys

from os import listdir, getcwd, makedirs
from os.path import abspath, exists, join, dirname

from rsmtool import VERSION_STRING
from rsmtool.analyzer import Analyzer
from rsmtool.configuration_parser import ConfigurationParser, Configuration
from rsmtool.modeler import Modeler
from rsmtool.preprocessor import FeaturePreprocessor
from rsmtool.reader import DataReader
from rsmtool.reporter import Reporter
from rsmtool.utils import LogFormatter
from rsmtool.writer import DataWriter


def run_experiment(config_file_or_obj,
                   output_dir):
    """
    Run RSMTool experiment using the given configuration
    file and generate all outputs in the given directory.

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

    # create the 'output' and the 'figure' sub-directories
    # where all the experiment output such as the CSV files
    # and the box plots will be saved

    # Get absolute paths to output directories
    csvdir = abspath(join(output_dir, 'output'))
    figdir = abspath(join(output_dir, 'figure'))
    reportdir = abspath(join(output_dir, 'report'))
    featuredir = abspath(join(output_dir, 'feature'))

    # Make directories, if necessary
    makedirs(csvdir, exist_ok=True)
    makedirs(figdir, exist_ok=True)
    makedirs(reportdir, exist_ok=True)

    # Allow users to pass Configuration object to the
    # `config_file_or_obj` argument, rather than read from file
    if not isinstance(config_file_or_obj, Configuration):

        # Instantiate configuration parser object
        parser = ConfigurationParser.get_configparser(config_file_or_obj)
        configuration = parser.read_normalize_validate_and_process_config(config_file_or_obj)

        # get the directory where the configuration file lives
        configpath = dirname(config_file_or_obj)

    else:

        configuration = config_file_or_obj
        if configuration.filepath is not None:
            configpath = dirname(configuration.filepath)
        else:
            configpath = getcwd()

    logger.info('Saving configuration file.')
    configuration.save(output_dir)

    # Get output format
    file_format = configuration.get('file_format', 'csv')

    # Get DataWriter object
    writer = DataWriter(configuration['experiment_id'])

    # Get the paths and names for the DataReader

    (file_names,
     file_paths_org) = configuration.get_names_and_paths(['train_file', 'test_file',
                                                          'features', 'feature_subset_file'],
                                                         ['train', 'test', 'feature_specs',
                                                          'feature_subset_specs'])

    file_paths = DataReader.locate_files(file_paths_org, configpath)

    # check that we were able to locate all files

    if None in file_paths:
        indices_with_no_paths = [i for i in range(len(file_paths))
                                 if file_paths[i] is None]

        missing_file_paths = [file_paths_org[i] for i in indices_with_no_paths]
        raise FileNotFoundError('The following files were not found: {}'.format(repr(missing_file_paths)))

    # Use the default converter for both train and test
    converters = {'train': configuration.get_default_converter(),
                  'test': configuration.get_default_converter()}

    logger.info('Reading in all data from files.')

    # Initialize the reader
    reader = DataReader(file_paths, file_names, converters)
    data_container = reader.read()

    logger.info('Preprocessing all features.')

    # Initialize the processor
    processor = FeaturePreprocessor()

    (processed_config,
     processed_container) = processor.process_data(configuration,
                                                   data_container)

    # Rename certain frames with more descriptive names
    # for writing out experiment files
    rename_dict = {'train_excluded': 'train_excluded_responses',
                   'test_excluded': 'test_excluded_responses',
                   'train_length': 'train_response_lengths',
                   'train_flagged': 'train_responses_with_excluded_flags',
                   'test_flagged': 'test_responses_with_excluded_flags'}

    logger.info('Saving training and test set data to disk.')

    # Write out files
    writer.write_experiment_output(csvdir,
                                   processed_container,
                                   ['train_features',
                                    'test_features',
                                    'train_metadata',
                                    'test_metadata',
                                    'train_other_columns',
                                    'test_other_columns',
                                    'train_preprocessed_features',
                                    'test_preprocessed_features',
                                    'train_excluded',
                                    'test_excluded',
                                    'train_length',
                                    'test_human_scores',
                                    'train_flagged',
                                    'test_flagged'],
                                   rename_dict,
                                   file_format=file_format)

    # Initialize the analyzer
    analyzer = Analyzer()

    (analyzed_config,
     analyzed_container) = analyzer.run_data_composition_analyses_for_rsmtool(processed_container,
                                                                              processed_config)

    # Write out files
    writer.write_experiment_output(csvdir,
                                   analyzed_container,
                                   file_format=file_format)

    logger.info('Training {} model.'.format(processed_config['model_name']))

    # Initialize modeler
    modeler = Modeler()

    modeler.train(processed_config,
                  processed_container,
                  csvdir,
                  figdir,
                  file_format)

    # Identify the features used by the model
    selected_features = modeler.get_feature_names()

    # Add selected features to processed configuration
    processed_config['selected_features'] = selected_features

    # Write out files
    writer.write_feature_csv(featuredir,
                             processed_container,
                             selected_features,
                             file_format=file_format)

    features_data_container = processed_container.copy()

    # Get selected feature info, and write out to file
    df_feature_info = features_data_container.feature_info.copy()
    df_selected_feature_info = df_feature_info[df_feature_info['feature'].isin(selected_features)]
    selected_feature_dataset_dict = {'name': 'selected_feature_info',
                                     'frame': df_selected_feature_info}

    features_data_container.add_dataset(selected_feature_dataset_dict,
                                        update=True)

    writer.write_experiment_output(csvdir,
                                   features_data_container,
                                   dataframe_names=['selected_feature_info'],
                                   new_names_dict={'selected_feature_info': 'feature'},
                                   file_format=file_format)

    logger.info('Running analyses on training set.')

    (train_analyzed_config,
     train_analyzed_container) = analyzer.run_training_analyses(processed_container,
                                                                processed_config)

    # Write out files
    writer.write_experiment_output(csvdir,
                                   train_analyzed_container,
                                   reset_index=True,
                                   file_format=file_format)

    # Use only selected features for predictions
    columns_for_prediction = ['spkitemid', 'sc1'] + selected_features
    train_for_prediction = processed_container.train_preprocessed_features[columns_for_prediction]
    test_for_prediction = processed_container.test_preprocessed_features[columns_for_prediction]

    logged_str = 'Generating training and test set predictions'
    logged_str += ' (expected scores).' if configuration['predict_expected_scores'] else '.'
    logger.info(logged_str)
    (pred_config,
     pred_data_container) = modeler.predict_train_and_test(train_for_prediction,
                                                           test_for_prediction,
                                                           processed_config)

    # Write out files
    writer.write_experiment_output(csvdir,
                                   pred_data_container,
                                   new_names_dict={'pred_test': 'pred_processed'},
                                   file_format=file_format)

    original_coef_file = join(csvdir, '{}_coefficients.{}'.format(pred_config['experiment_id'],
                                                                  file_format))

    # If coefficients file exists, then generate
    # scaled coefficients and save to file
    if exists(original_coef_file):
        logger.info('Scaling the coefficients and saving them to disk')
        try:

            # Scale coefficients, and return DataContainer w/ scaled coefficients
            scaled_data_container = modeler.scale_coefficients(pred_config)

            # Write out files to disk
            writer.write_experiment_output(csvdir,
                                           scaled_data_container,
                                           file_format=file_format)

        except AttributeError:
            raise ValueError("It appears you are trying to save two different "
                             "experiments to the same directory using the same "
                             "ID. Please clear the content of the directory and "
                             "rerun both experiments using different "
                             "experiment IDs.")

    # Add processed data_container frames to pred_data_container
    new_pred_data_container = pred_data_container + processed_container

    logger.info('Running prediction analyses.')
    (pred_analysis_config,
     pred_analysis_data_container) = analyzer.run_prediction_analyses(new_pred_data_container,
                                                                      pred_config)

    # Write out files
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
                           figdir)


def main():

    formatter = LogFormatter()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)

    logger = logging.getLogger(__name__)

    # set up an argument parser
    parser = argparse.ArgumentParser(prog='rsmtool')

    parser.add_argument('-f', '--force', dest='force_write',
                        action='store_true', default=False,
                        help="If true, rsmtool will not check if the "
                             "output directory already contains the "
                             "output of another rsmtool experiment. ")

    parser.add_argument('config_file', help="The JSON configuration file for "
                                            "this experiment")

    parser.add_argument('output_dir', nargs='?', default=getcwd(),
                        help="The output directory where all the files "
                             "for this experiment will be stored")

    parser.add_argument('-V', '--version', action='version',
                        version=VERSION_STRING)

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
    run_experiment(config_file, output_dir)


if __name__ == '__main__':

    main()
