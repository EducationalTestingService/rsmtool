"""
Utility to generate predictions on new data
from existing RSMTool models.

:author: Nitin Madnani (nmadnani@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:organization: ETS
"""

#!/usr/bin/env python

import argparse
import glob
import logging
import os
import sys

from os.path import basename, dirname, exists, join, normpath, splitext

import numpy as np
import pandas as pd

from rsmtool.input import (check_main_config,
                           locate_file,
                           read_data_file,
                           read_json_file,
                           rename_default_columns,
                           check_flag_column)
from rsmtool.predict import predict_with_model, process_predictions
from rsmtool.preprocess import (preprocess_new_data,
                                trim)
from rsmtool.utils import LogFormatter

from skll import Learner
from rsmtool.version import __version__



def compute_and_save_predictions(config_file, output_file, feats_file):
    """
    Run ``rsmpredict`` with given configuration file and generate
    predictions (and, optionally, pre-processed feature values).

    Parameters
    ----------
    config_file : str
        Path to the experiment configuration file.
    output_file : str
        Path to the output file for saving predictions.
    feats_file (optional): str
        Path to the output file for saving preprocessed feature values.

    Raises
    ------
    ValueError
        If any of the required fields are missing or ill-specified.

    """

    logger = logging.getLogger(__name__)

    # read in the main config file
    config_obj = read_json_file(config_file)
    config_obj = check_main_config(config_obj, context='rsmpredict')

    # get the directory where the config file lives
    # if this is the 'expm' directory, then go
    # up one level.
    configpath = dirname(config_file)

    # get the input file containing the feature values
    # for which we want to generate the predictions
    input_features_file = locate_file(config_obj['input_features_file'], configpath)
    if not input_features_file:
        raise FileNotFoundError('Input file {} does not exist'.format(config_obj['input_features_file']))

    # get the experiment ID
    experiment_id = config_obj['experiment_id']

    # get the column name that will hold the ID
    id_column = config_obj['id_column']

    # get the column name for human score (if any)
    human_score_column = config_obj['human_score_column']

    # get the column name for second human score (if any)
    second_human_score_column = config_obj['second_human_score_column']

    # get the column name for subgroups (if any)
    subgroups = config_obj['subgroups']

    # get the column names for flag columns (if any)
    flag_column_dict = check_flag_column(config_obj)

    # get the name for the candidate_column (if any)
    candidate_column = config_obj['candidate_column']

    # get the directory of the experiment
    experiment_dir = locate_file(config_obj['experiment_dir'], configpath)
    if not experiment_dir:
        raise FileNotFoundError('The directory {} does not exist.'.format(config_obj['experiment_dir']))
    else:
        experiment_output_dir = normpath(join(experiment_dir, 'output'))
        if not exists(experiment_output_dir):
            raise FileNotFoundError('The directory {} does not contain '
                                    'the output of an rsmtool experiment.'.format(experiment_dir))

    # find all the .model files in the experiment output directory
    model_files = glob.glob(join(experiment_output_dir, '*.model'))
    if not model_files:
        raise FileNotFoundError('The directory {} does not contain any rsmtool models.'.format(experiment_output_dir))

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

    # read in the given features but make sure that the
    # `id_column`, `candidate_column` and subgroups are read in as a string
    logger.info('Reading features from {}'.format(input_features_file))
    string_columns = [id_column, candidate_column] + subgroups
    converter_dict = dict([(column, str) for column in string_columns if column])

    df_input = read_data_file(input_features_file, converters=converter_dict)

    # make sure that the columns specified in the config file actually exist
    columns_to_check = [id_column] + subgroups + list(flag_column_dict.keys())

    # add subgroups and the flag columns to the list of columns
    # that will be added to the final file
    columns_to_copy = subgroups + list(flag_column_dict.keys())

    # human_score_column will be set to sc1 by default
    # we only raise an error if it's set to something else.
    # However, since we cannot distinguish whether the column was set
    # to sc1 by default or specified as such in the config file
    # we append it to output anyway as long as
    # it is in the input file

    if human_score_column != 'sc1' or 'sc1' in df_input.columns:
        columns_to_check.append(human_score_column)
        columns_to_copy.append('sc1')

    if candidate_column:
        columns_to_check.append(candidate_column)
        columns_to_copy.append('candidate')

    if second_human_score_column:
        columns_to_check.append(second_human_score_column)
        columns_to_copy.append('sc2')

    missing_columns = set(columns_to_check).difference(df_input.columns)
    if missing_columns:
        raise KeyError("Columns {} from the config file "
                       "do not exist in the data.".format(missing_columns))

    # rename all columns
    df_input = rename_default_columns(df_input,
                                      [],
                                      id_column,
                                      human_score_column,
                                      second_human_score_column,
                                      None,
                                      None,
                                      candidate_column=candidate_column)

    # check that the id_column contains unique values
    if df_input['spkitemid'].size != df_input['spkitemid'].unique().size:
        raise ValueError("The data contains repeated response IDs in {}. Please make sure all response IDs are unique and re-run the tool.".format(id_column))

    #  Read the preprocessing parameters stored in the
    # _features.csv file
    df_feature_info = pd.read_csv(join(experiment_output_dir,
                                       '{}_feature.csv'.format(experiment_id)),
                                  index_col=0)

    (df_features_preprocessed,
                  df_excluded) = preprocess_new_data(df_input,
                                                     df_feature_info)

    # save the pre-processed features to disk if we were asked to
    if feats_file:
        logger.info('Saving pre-processed feature values to {}'.format(feats_file))

        # create any directories needed for the output file
        os.makedirs(dirname(feats_file), exist_ok=True)
        df_features_preprocessed.to_csv(feats_file, index=False)

    # now load the SKLL model to generate the predictions
    model = Learner.from_file(join(experiment_output_dir, '{}.model'.format(experiment_id)))

    # now generate the predictions for the features using this model
    logger.info('Generating predictions')
    df_predictions = predict_with_model(model, df_features_preprocessed)

    # read in the post-processing parameters from disk
    df_postproc_params = pd.read_csv(join(experiment_output_dir, '{}_postprocessing_params.csv'.format(experiment_id)))

    trim_min = df_postproc_params['trim_min'].values[0]
    trim_max = df_postproc_params['trim_max'].values[0]
    h1_mean = df_postproc_params['h1_mean'].values[0]
    h1_sd = df_postproc_params['h1_sd'].values[0]
    train_predictions_mean = df_postproc_params['train_predictions_mean'].values[0]
    train_predictions_sd = df_postproc_params['train_predictions_sd'].values[0]

    df_predictions = process_predictions(df_predictions,
                                         train_predictions_mean,
                                         train_predictions_sd,
                                         h1_mean,
                                         h1_sd,
                                         trim_min, trim_max)

    # add back the columns that we were requested to copy if any
    if len(columns_to_copy) > 0:
        df_predictions_with_metadata = pd.merge(df_predictions,
                                                df_input[['spkitemid'] + columns_to_copy])
        assert(len(df_predictions) == len(df_predictions_with_metadata))
    else:
        df_predictions_with_metadata = df_predictions.copy()

    # create any directories needed for the output file
    os.makedirs(dirname(output_file), exist_ok=True)

    # save the predictions to disk
    logger.info('Saving predictions to {}'.format(output_file))
    df_predictions_with_metadata.to_csv(output_file, index=False)

    # save excluded responses to disk
    if not df_excluded.empty:
        excluded_output_file = '{}_excluded_responses{}'.format(*splitext(output_file))
        logger.info('Saving excluded responses to {}'.format(excluded_output_file))
        df_excluded.to_csv(excluded_output_file, index=False)


def main():

    # set up the basic logging config
    fmt = LogFormatter()
    hdlr = logging.StreamHandler(sys.stdout)
    hdlr.setFormatter(fmt)
    logging.root.addHandler(hdlr)
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
                        version='%(prog)s {0}'.format(__version__))

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
