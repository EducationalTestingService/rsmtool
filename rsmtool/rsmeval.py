"""
Run evaluation only experiments.

:author: Nitin Madnani (nmadnani@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:organization: ETS
"""

#!/usr/bin/env python

import argparse
import json
import logging
import os
import sys

from os import listdir
from os.path import abspath, dirname, exists, join

import numpy as np
import pandas as pd

from rsmtool.analysis import (run_data_composition_analyses_for_rsmeval,
                              run_prediction_analyses)

from rsmtool.input import (check_main_config,
                           check_subgroups,
                           get_trim_min_max,
                           locate_file,
                           read_json_file,
                           rename_default_columns,
                           check_flag_column,
                           locate_custom_sections,
                           select_candidates_with_N_or_more_items)

from rsmtool.predict import process_predictions

from rsmtool.preprocess import (filter_on_column,
                                filter_on_flag_columns)

from rsmtool.report import (create_report,
                            get_ordered_notebook_files)

from rsmtool.utils import write_experiment_output

from rsmtool.utils import LogFormatter

from rsmtool.version import __version__


def run_evaluation(config_file, output_dir):
    """
    Run an `rsmeval` experiment using the given configuration
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

    # create the 'output' and the 'figure' sub-directories
    # where all the experiment output such as the CSV files
    # and the box plots will be saved
    csvdir = abspath(join(output_dir, 'output'))
    figdir = abspath(join(output_dir, 'figure'))
    reportdir = abspath(join(output_dir, 'report'))
    os.makedirs(csvdir, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(reportdir, exist_ok=True)

    # load the information from the config file
    # read in the main config file
    config_obj = read_json_file(config_file)
    config_obj = check_main_config(config_obj, context='rsmeval')

    # save a copy of the normalized config into the experiment directory
    outjson = join(output_dir, 'output', '{}_rsmeval.json'.format(config_obj['experiment_id']))
    with open(outjson, 'w') as outfile:
        json.dump(config_obj, outfile, indent=4, separators=(',', ': '))

    # get the directory where the config file lives
    # if this is the 'expm' directory, then go
    # up one level.
    configpath = dirname(config_file)

    # get the experiment ID
    experiment_id = config_obj['experiment_id']

    # get the description
    description = config_obj['description']

    # get the column name for the labels for the training and testing data
    human_score_column = config_obj['human_score_column']
    system_score_column = config_obj['system_score_column']

    # if the human score column is the same as the
    # system score column, raise an error
    if human_score_column == system_score_column:
        raise ValueError("'human_score_column' and "
                         "'system_score_column' "
                         "cannot have the same value.")

    # get the name of the optional column that
    # contains the second human score
    second_human_score_column = config_obj['second_human_score_column']

    # if the human score column is the same as the
    # second human score column, raise an error
    if human_score_column == second_human_score_column:
        raise ValueError("'human_score_column' and "
                         "'second_human_score_column' "
                         "cannot have the same value.")

    # get the column name that will hold the ID for
    # both the training and the test data
    id_column = config_obj['id_column']

    # get the specified trim min and max, if any
    # and make sure they are numeric
    spec_trim_min, spec_trim_max = get_trim_min_max(config_obj)

    # get the subgroups if any
    subgroups = config_obj.get('subgroups')

    # get the candidate column if any and convert it to string
    candidate_column = config_obj['candidate_column']

    # check if we are excluding candidates based on number of responses
    exclude_listwise = False
    min_items_per_candidate = config_obj['min_items_per_candidate']
    if min_items_per_candidate:
        exclude_listwise = True

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

    #  check all sections values and order and get the
    # ordered list of notebook files
    chosen_notebook_files = get_ordered_notebook_files(general_report_sections,
                                                       special_report_sections,
                                                       custom_report_sections,
                                                       section_order,
                                                       subgroups,
                                                       model_type=None,
                                                       context='rsmeval')
    # are we excluding zero scores?
    exclude_zero_scores = config_obj['exclude_zero_scores']

    # if we are excluding zero scores but trim_min
    # is set to 0, then we need to warn the user
    if exclude_zero_scores and spec_trim_min == 0:
        logger.warning("'exclude_zero_scores' is set to True but "
                       " 'trim_min' is set to 0. This may cause "
                       " unexpected behavior.")

    # are we filtering on any other columns?
    flag_column_dict = check_flag_column(config_obj)

    # do we have the training set predictions and human scores CSV file
    scale_with = config_obj.get('scale_with')

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

    # we need to scale if and only if a CSV file is specified
    do_scaling = (scale_with is not None and scale_with != 'asis')

    # use scaled predictions for the analyses unless
    # we were told not to
    use_scaled_predictions = (scale_with is not None)

    # log an appropriate message
    if scale_with is None:
        message = ('Assuming given system predictions '
                   'are unscaled and will be used as such.')
    elif scale_with == 'asis':
        message = ('Assuming given system predictions '
                   'are already scaled and will be used as such.')
    else:
        message = ('Assuming given system predictions '
                   'are unscaled and will be scaled before use.')
    logger.info(message)

    # load the predictions from disk and make sure that the `id_column`
    # is read in as a string
    predictions_file_location = locate_file(config_obj['predictions_file'], configpath)
    if not predictions_file_location:
        raise FileNotFoundError('Error: Predictions file {} '
                                'not found.\n'.format(config_obj['predictions_file']))
    else:
        logger.info('Reading predictions: {}'.format(predictions_file_location))
        string_columns = [id_column, candidate_column] + subgroups
        converter_dict = dict([(column, str) for column in string_columns if column])

        df_pred = pd.read_csv(predictions_file_location, converters=converter_dict)

    # make sure that the columns specified in the config file actually exist

    # make sure that the columns specified in the config file actually exist
    columns_to_check = [id_column, human_score_column, system_score_column]

    if second_human_score_column:
        columns_to_check.append(second_human_score_column)

    if candidate_column:
        columns_to_check.append(candidate_column)

    missing_columns = set(columns_to_check).difference(df_pred.columns)
    if missing_columns:
        raise KeyError('Columns {} from the config file do not exist '
                       'in the predictions file.'.format(missing_columns))

    df_pred = rename_default_columns(df_pred,
                                     [],
                                     id_column,
                                     human_score_column,
                                     second_human_score_column,
                                     None,
                                     system_score_column,
                                     candidate_column)

    # check that the id_column contains unique values
    if df_pred['spkitemid'].size != df_pred['spkitemid'].unique().size:
        raise ValueError("The data contains duplicate response IDs "
                         "in '{}'. Please make sure all response IDs "
                         "are unique and re-run the tool.".format(id_column))

    df_pred = check_subgroups(df_pred, subgroups)

    # filter out the responses based on flag columns
    (df_responses_with_requested_flags,
     df_responses_with_excluded_flags) = filter_on_flag_columns(df_pred, flag_column_dict)

    # filter out rows that have non-numeric or zero human scores
    df_filtered, df_excluded = filter_on_column(df_responses_with_requested_flags,
                                                'sc1',
                                                'spkitemid',
                                                exclude_zeros=exclude_zero_scores)

    # make sure that the remaining data frame is not empty
    if len(df_filtered) == 0:
        raise ValueError("No responses remaining after filtering out "
                         "non-numeric human scores. No further analysis "
                         "can be run. ")

    # Change all non-numeric machine scores in excluded
    # data to NaNs for consistency with rsmtool.
    # NOTE: This will *not* work if *all* of the values
    # in column are non-numeric. This is a known bug in
    # pandas: https://github.com/pydata/pandas/issues/9589
    # Therefore, we need add an additional check after this.
    df_excluded['raw'] = pd.to_numeric(df_excluded['raw'], errors='coerce').astype(float)

    # filter out the non-numeric machine scores from the rest of the data
    newdf, newdf_excluded = filter_on_column(df_filtered,
                                             'raw',
                                             'spkitemid',
                                             exclude_zeros=False)

    del df_filtered
    df_filtered_pred = newdf

    # make sure that the remaining data frame is not empty
    if len(df_filtered_pred) == 0:
        raise ValueError("No responses remaining after filtering out "
                         "non-numeric machine scores. No further analysis "
                         "can be run. ")

    df_excluded = pd.merge(df_excluded, newdf_excluded, how='outer')

    # if requested, exclude the candidates with less than X responses
    # left after filtering
    if exclude_listwise:
        (df_filtered_candidates,
         df_excluded_candidates) = select_candidates_with_N_or_more_items(df_filtered_pred,
                                                                          min_items_per_candidate)
        # check that there are still responses left for analysis
        if len(df_filtered_candidates) == 0:
            raise ValueError("After filtering non-numeric human and system scores "
                             "there were "
                             "no candidates with {} or more responses "
                             "left for analysis".format(str(min_items_per_candidate)))

        # redefine df_filtered_pred
        df_filtered_pred = df_filtered_candidates.copy()

        # update df_excluded
        df_excluded = pd.concat([df_excluded, df_excluded_candidates])

    # set default values for scaling
    scale_pred_mean = 0
    scale_pred_sd = 1
    scale_human_mean = 0
    scale_human_sd = 1

    if do_scaling:
        scale_file_location = locate_file(scale_with, configpath)
        if not scale_file_location:
            raise FileNotFoundError('Error: scaling file {} not found.\n'.format(scale_with))
        else:
            logger.info('Reading scaling file: {}'.format(scale_file_location))
            df_scale_with = pd.read_csv(scale_file_location)

        if 'sc1' not in df_scale_with.columns and 'prediction' not in df_scale_with.columns:
            raise KeyError('The CSV file specified for scaling ',
                           'must have the "prediction" and the "sc1" '
                           'columns.')
        else:
            scale_pred_mean, scale_pred_sd = (df_scale_with['prediction'].mean(),
                                              df_scale_with['prediction'].std())
            scale_human_mean, scale_human_sd = (df_scale_with['sc1'].mean(),
                                                df_scale_with['sc1'].std())

    logger.info('Processing predictions')
    df_pred_processed = process_predictions(df_filtered_pred,
                                            scale_pred_mean,
                                            scale_pred_sd,
                                            scale_human_mean,
                                            scale_human_sd,
                                            spec_trim_min, spec_trim_max)
    if not scale_with:
        expected_score_types = ['raw', 'raw_trim', 'raw_trim_round']
    elif scale_with == 'asis':
        expected_score_types = ['scale', 'scale_trim', 'scale_trim_round']
    else:
        expected_score_types = ['raw', 'raw_trim', 'raw_trim_round', 'scale', 'scale_trim', 'scale_trim_round']

    # extract separated data frames that we will write out
    # as separate files
    not_other_columns = set()

    prediction_columns = ['spkitemid', 'sc1'] + expected_score_types
    df_predictions_only = df_pred_processed[prediction_columns]
    not_other_columns.update(prediction_columns)

    metadata_columns = ['spkitemid'] + subgroups
    if candidate_column:
        metadata_columns.append('candidate')
    df_test_metadata = df_filtered_pred[metadata_columns]
    not_other_columns.update(metadata_columns)

    df_test_human_scores = pd.DataFrame()
    human_score_columns = ['spkitemid', 'sc1', 'sc2']
    if second_human_score_column and 'sc2' in df_filtered_pred:
        df_test_human_scores = df_filtered_pred[human_score_columns].copy()
        not_other_columns.update(['sc2'])
        # filter out any non-numeric values nows
        # as well as zeros, if we were asked to
        df_test_human_scores['sc2'] = pd.to_numeric(df_test_human_scores['sc2'],
                                                    errors='coerce').astype(float)
        if exclude_zero_scores:
            df_test_human_scores['sc2'] = df_test_human_scores['sc2'].replace(0, np.nan)

    # remove 'spkitemid' from `not_other_columns`
    # because we want that in the other columns
    # data frame
    not_other_columns.remove('spkitemid')

    # extract all of the other columns in the predictions file
    other_columns = [column for column in df_filtered_pred.columns
                     if column not in not_other_columns]
    df_pred_other_columns = df_filtered_pred[other_columns]

    logger.info('Saving pre-processed predictions and the metadata to disk')
    write_experiment_output([df_predictions_only,
                             df_test_metadata,
                             df_pred_other_columns,
                             df_test_human_scores,
                             df_excluded,
                             df_responses_with_excluded_flags],
                            ['pred_processed',
                             'test_metadata',
                             'test_other_columns',
                             'test_human_scores',
                             'test_excluded_responses',
                             'test_responses_with_excluded_flags'],
                            experiment_id,
                            csvdir)

    # do the data composition stats
    (df_test_excluded_analysis,
     df_data_composition,
     data_composition_by_group_dict) = run_data_composition_analyses_for_rsmeval(df_test_metadata,
                                                                                 df_excluded,
                                                                                 subgroups,
                                                                                 candidate_column,
                                                                                 exclude_zero_scores=exclude_zero_scores,
                                                                                 exclude_listwise=exclude_listwise)

    write_experiment_output([df_test_excluded_analysis,
                             df_data_composition],
                            ['test_excluded_composition',
                             'data_composition'],
                            experiment_id,
                            csvdir)

    # write the results of data composition analysis by group
    if subgroups:
        for group in subgroups:
            write_experiment_output([data_composition_by_group_dict[group]], ['data_composition_by_{}'.format(group)], experiment_id, csvdir)

    # run the analyses on the predictions of the modelx`
    logger.info('Running analyses on predictions')
    (df_human_machine_eval,
     df_human_machine_eval_short,
     df_human_human_eval,
     eval_by_group_dict,
     df_degradation,
     df_confmatrix,
     df_score_dist) = run_prediction_analyses(df_predictions_only,
                                              df_test_metadata,
                                              df_test_human_scores,
                                              subgroups,
                                              second_human_score_column,
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

    # if we are using subgroups, then write out the subgroup
    # specific output and include the by group section
    # in the final report
    if subgroups:
        for group in subgroups:
            eval_by_group, consistency_by_group = eval_by_group_dict[group]
            write_experiment_output([eval_by_group, consistency_by_group],
                                    ['eval_by_{}'.format(group),
                                     'consistency_by_{}'.format(group)],
                                    experiment_id,
                                    csvdir,
                                    reset_index=True)

    # generate the report
    logger.info('Starting report generation')
    create_report(experiment_id, description,
                  '', '',
                  '', predictions_file_location,
                  csvdir, figdir,
                  subgroups,
                  None,
                  second_human_score_column,
                  min_items_per_candidate,
                  chosen_notebook_files,
                  exclude_zero_scores=exclude_zero_scores,
                  use_scaled_predictions=use_scaled_predictions,
                  context='rsmeval')


def main():

    # set up the basic logging config
    fmt = LogFormatter()
    hdlr = logging.StreamHandler(sys.stdout)
    hdlr.setFormatter(fmt)
    logging.root.addHandler(hdlr)
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

    parser.add_argument('config_file', help="The JSON config file for "
                                            "this experiment")

    parser.add_argument('output_dir', nargs='?', default=os.getcwd(),
                        help="The output directory where all the files "
                             "for this experiment will be stored")

    parser.add_argument('-V', '--version', action='version',
                        version='%(prog)s {0}'.format(__version__))

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

    # make sure that the given config file exists
    if not exists(config_file):
        raise FileNotFoundError("Main config file {} not "
                                "found.".format(config_file))

    # run the evaluation experiment
    run_evaluation(config_file, output_dir)

if __name__ == '__main__':
    main()
