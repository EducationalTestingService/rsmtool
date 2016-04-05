"""
The main RSMTool script.

:author: Nitin Madnani (nmadnani@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:organization: ETS
"""

#!/usr/bin/env python

import argparse
import logging
import os
import sys

from os import listdir
from os.path import abspath, exists, join

import pandas as pd

from rsmtool.analysis import (run_training_analyses,
                              run_prediction_analyses,
                              run_data_composition_analyses_for_rsmtool)
from rsmtool.input import load_experiment_data
from rsmtool.model import train_model
from rsmtool.predict import generate_train_and_test_predictions
from rsmtool.preprocess import preprocess_train_and_test_features
from rsmtool.report import create_report
from rsmtool.utils import (scale_coefficients,
                           write_experiment_output,
                           write_feature_json)
from rsmtool.utils import LogFormatter

def run_experiment(config_file, output_dir):
    """
    Run RSMTool experiment using the given configuration
    file and generate all outputs in the given directory.
    """

    logger = logging.getLogger(__name__)

    # create the 'output' and the 'figure' sub-directories
    # where all the experiment output such as the CSV files
    # and the box plots will be saved
    csvdir = abspath(join(output_dir, 'output'))
    figdir = abspath(join(output_dir, 'figure'))
    reportdir = abspath(join(output_dir, 'report'))
    featuredir = abspath(join(output_dir, 'feature'))
    os.makedirs(csvdir, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(reportdir, exist_ok=True)

    # load the experiment data
    logger.info('Loading experiment data')
    (df_train_features, df_test_features,
     df_train_metadata, df_test_metadata,
     df_train_other_columns, df_test_other_columns,
     df_train_excluded, df_test_excluded,
     df_train_length, df_test_human_scores,
     df_train_flagged_responses,
     df_test_flagged_responses,
     experiment_id, description,
     train_file_location, test_file_location,
     feature_specs, model_name, model_type,
     train_label_column, test_label_column,
     id_column, length_column, second_human_score_column,
     candidate_column, subgroups,
     feature_subset_file,
     used_trim_min, used_trim_max,
     use_scaled_predictions,
     exclude_zero_scores,
     select_features_automatically,
     chosen_notebook_files) = load_experiment_data(config_file, output_dir)

    # preprocess each feature for the training and testing data
    logger.info('Pre-processing training and test set features')
    (df_train_preprocessed_features,
     df_test_preprocessed_features,
     df_feature_info) = preprocess_train_and_test_features(df_train_features,
                                                           df_test_features,
                                                           feature_specs)

    logger.info('Saving training and test set data to disk')
    write_experiment_output([df_train_features, df_test_features,
                             df_train_metadata,
                             df_test_metadata,
                             df_train_other_columns,
                             df_test_other_columns,
                             df_train_preprocessed_features,
                             df_test_preprocessed_features,
                             df_train_excluded,
                             df_test_excluded,
                             df_train_length,
                             df_test_human_scores,
                             df_train_flagged_responses,
                             df_test_flagged_responses],
                            ['train_features',
                             'test_features',
                             'train_metadata',
                             'test_metadata',
                             'train_other_columns',
                             'test_other_columns',
                             'train_preprocessed_features',
                             'test_preprocessed_features',
                             'train_excluded_responses',
                             'test_excluded_responses',
                             'train_response_lengths',
                             'test_human_scores',
                             'train_responses_with_excluded_flags',
                             'test_responses_with_excluded_flags'],
                            experiment_id,
                            csvdir)

    # do the data composition stats
    features = [column for column in df_train_features.columns if column not in ['spkitemid', 'sc1']]
    (df_train_excluded_analysis,
     df_test_excluded_analysis,
     df_data_composition,
     data_composition_by_group_dict) = run_data_composition_analyses_for_rsmtool(df_train_metadata,
                                                                                 df_test_metadata,
                                                                                 df_train_excluded,
                                                                                 df_test_excluded,
                                                                                 features,
                                                                                 subgroups,
                                                                                 candidate_column,
                                                                                 exclude_zero_scores=exclude_zero_scores)
    write_experiment_output([df_train_excluded_analysis,
                             df_test_excluded_analysis,
                             df_data_composition],
                            ['train_excluded_composition',
                             'test_excluded_composition',
                             'data_composition'],
                            experiment_id,
                            csvdir)

    # write the results of data composition analysis by group
    for group in subgroups:
        write_experiment_output([data_composition_by_group_dict[group]], ['data_composition_by_{}'.format(group)], experiment_id, csvdir)

    # train the appropriate model. This is done before the
    # descriptive analyses since for models with feature
    # selection we only do the analysis for the
    # features selected for the final model
    logger.info('Training {} model'.format(model_name))
    model = train_model(model_name,
                        df_train_preprocessed_features,
                        experiment_id,
                        csvdir,
                        figdir)

    # identify the features used by the model
    selected_features = model.feat_vectorizer.get_feature_names()

    # if this is not the same set as what was originally
    # specified or no set was specified
    # if we had a subset of features from the user, check if the
    # final subset of features matches that and if not set the feature
    # selection to automatic
    if not select_features_automatically:
        requested_features = df_feature_info['feature'].tolist()
        omitted_features = set(requested_features).difference(selected_features)
        if omitted_features:
            select_features_automatically = True

    # for all models with automatic feature selection
    # generate a .json file with selected features.
    # this can be used to train models using just this selection.

    if select_features_automatically:
        write_feature_json(feature_specs, selected_features, experiment_id, featuredir)

    df_selected_feature_info = df_feature_info[df_feature_info['feature'].isin(selected_features)]
    write_experiment_output([df_selected_feature_info],
                            ['feature'],
                            experiment_id,
                            csvdir)

    # run the training set analyses
    logger.info('Running analyses on training set')

    (df_descriptives,
     df_percentiles,
     df_outliers,
     df_all_pairwise_cors_orig,
     df_all_pairwise_cors,
     df_margcor_sc1,
     df_pcor_sc1,
     df_pcor_sc1_no_length,
     df_margcor_length,
     df_pcor_length,
     score_correlation_by_group_dict,
     length_correlation_by_group_dict,
     df_pca_components,
     df_pca_variance) = run_training_analyses(df_train_features,
                                              df_train_metadata,
                                              df_train_preprocessed_features,
                                              df_train_length,
                                              length_column,
                                              selected_features,
                                              subgroups)

    write_experiment_output([df_descriptives,
                             df_percentiles,
                             df_outliers,
                             df_all_pairwise_cors_orig,
                             df_all_pairwise_cors,
                             df_margcor_sc1, df_pcor_sc1,
                             df_pcor_sc1_no_length,
                             df_margcor_length, df_pcor_length,
                             df_pca_components,
                             df_pca_variance],
                            ['feature_descriptives',
                             'feature_descriptivesExtra',
                             'feature_outliers',
                             'cors_orig',
                             'cors_processed',
                             'margcor_score_all_data',
                             'pcor_score_all_data',
                             'pcor_score_no_length_all_data',
                             'margcor_length_all_data',
                             'pcor_length_all_data',
                             'pca', 'pcavar'],
                            experiment_id,
                            csvdir,
                            reset_index=True)

    # write the results of score and length correlation analyses by group
    for group in subgroups:
        sc1_marg_cors, sc1_part_cors, sc1_part_cors_no_length = score_correlation_by_group_dict[group]
        write_experiment_output([sc1_marg_cors, sc1_part_cors, sc1_part_cors_no_length],
                                ['margcor_score_by_{}'.format(group),
                                 'pcor_score_by_{}'.format(group),
                                 'pcor_score_no_length_by_{}'.format(group)],
                                experiment_id,
                                csvdir,
                                reset_index=True)

        length_marg_cors, length_part_cors, _ = length_correlation_by_group_dict.get(group,
                                                                                  (pd.DataFrame(),
                                                                                   pd.DataFrame(),
                                                                                   pd.DataFrame()))
        write_experiment_output([length_marg_cors, length_part_cors],
                                ['margcor_length_by_{}'.format(group),
                                 'pcor_length_by_{}'.format(group)],
                                experiment_id,
                                csvdir,
                                reset_index=True)

    # now generate the predictions using this model on the training
    # and testing data
    logger.info('Generating training and test set predictions')

    # Tell the user that they will get the mismatch warning if
    # the model features do not match the original data

    if len(selected_features) != len(df_test_preprocessed_features.columns)-2:
        logger.warning("You specified a model with automatic feature "
                       "selection and therefore some of the original "
                       "features were excluded from the final model. If this "
                       "is the expected behavior, you can ignore "
                       "the following warnings about the mismatch in the data")

    (df_train_predictions,
     df_test_predictions,
     train_predictions_mean,
     train_predictions_sd,
     h1_mean,
     h1_sd) = generate_train_and_test_predictions(model,
                                                  df_train_preprocessed_features,
                                                  df_test_preprocessed_features,
                                                  used_trim_min,
                                                  used_trim_max)

    # create a data frame with the post-processing parameters
    # we want to save to disk
    df_postproc_params = pd.DataFrame([{'trim_min': used_trim_min,
                                        'trim_max': used_trim_max,
                                        'h1_mean': h1_mean,
                                        'h1_sd': h1_sd,
                                        'train_predictions_mean': train_predictions_mean,
                                        'train_predictions_sd': train_predictions_sd}])

    logger.info('Saving training and test set predictions to disk')
    write_experiment_output([df_train_predictions,
                             df_test_predictions,
                             df_postproc_params],
                            ['pred_train',
                             'pred_processed',
                             'postprocessing_params'],
                            experiment_id,
                            csvdir)

    # scale coefficients using the predictions and save them
    # into a separate file -
    # we only do this for models which generate _coefficients file.

    original_coef_file = join(csvdir, "{}_coefficients.csv".format(experiment_id))

    if exists(original_coef_file):
        logger.info('Scaling the coefficients and saving them to disk')
        try:
            df_scaled_coefficients = scale_coefficients(model.model.intercept_,
                                                        model.model.coef_,
                                                        model.feat_vectorizer.get_feature_names(),
                                                        train_predictions_mean,
                                                        train_predictions_sd,
                                                        h1_sd)
        except AttributeError:
            raise ValueError("It appears you are trying to save two different "
                             "experiments to the same directory using the same "
                             "ID. Please clear the content of the directory and "
                             "rerun both experiments using different "
                             "experiment IDs.")
        write_experiment_output([df_scaled_coefficients],
                                ['coefficients_scaled'],
                                experiment_id,
                                csvdir)

    # run the analyses on the predictions of the models

    logger.info('Running analyses on test set predictions')
    (df_human_machine_eval,
     df_human_machine_eval_short,
     df_human_human_eval,
     eval_by_group_dict,
     df_degradation,
     df_confmatrix,
     df_score_dist) = run_prediction_analyses(df_test_predictions,
                                              df_test_metadata,
                                              df_test_human_scores,
                                              subgroups,
                                              second_human_score_column,
                                              exclude_zero_scores=exclude_zero_scores,
                                              use_scaled_predictions=use_scaled_predictions)

    write_experiment_output([df_human_machine_eval,
                             df_human_machine_eval_short,
                             df_human_human_eval,
                             df_degradation,
                             df_confmatrix,
                             df_score_dist],
                            ['eval',
                             'eval_short',
                             'consistency',
                             'degradation',
                             'confMatrix',
                             'score_dist'],
                            experiment_id,
                            csvdir,
                            reset_index=True)

    # write the results of evaluations by group
    for group in subgroups:
        eval_by_group, consistency_by_group = eval_by_group_dict[group]
        write_experiment_output([eval_by_group, consistency_by_group],
                                ['eval_by_{}'.format(group),
                                 'consistency_by_{}'.format(group)],
                                experiment_id, csvdir, reset_index=True)

    # generate the report
    logger.info('Starting report generation')
    create_report(experiment_id, description,
                  model_type, model_name,
                  train_file_location,
                  test_file_location,
                  csvdir, figdir,
                  subgroups,
                  length_column,
                  second_human_score_column,
                  feature_subset_file=feature_subset_file,
                  chosen_notebook_files=chosen_notebook_files,
                  exclude_zero_scores=exclude_zero_scores,
                  use_scaled_predictions=use_scaled_predictions)

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
    parser = argparse.ArgumentParser(prog='rsmtool.py')

    parser.add_argument('-f', '--force', dest='force_write',
                        action='store_true', default=False,
                        help="If true, rsmtool will not check if the"
                             " output directory already contains the "
                             "output of another rsmtool experiment. ")

    parser.add_argument('config_file', help="The JSON config file for "
                                            "this experiment")

    parser.add_argument('output_dir', nargs='?', default=os.getcwd(),
                        help="The output directory where all the files "
                             "for this experiment will be stored")

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

    # make sure that the given config file exists
    if not exists(config_file):
        raise FileNotFoundError('Main config file {} '
                                'not found.'.format(config_file))

    # run the experiment
    run_experiment(config_file, output_dir)


if __name__ == '__main__':
    main()

