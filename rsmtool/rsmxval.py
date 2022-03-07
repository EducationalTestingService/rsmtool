#!/usr/bin/env python

"""
Run a cross-validation experiment using RSMTool, RSMEval, and RSMSummarize.

The complete cross-validation workflow is as follows:

- Divide the training data file into folds either by randomly shuffling
  and then using a k-fold split or by using a pre-specified folds file
  that provides a fold number for each example ID in the data.

- For each fold (in parallel), use rsmtool to run a train/test experiment
  and save all output in a separate directory for that fold under
  `<outdir>/folds/<num>`.

- Combine the predictions for each fold's test set and run rsmeval on that
  file. Save all output under `<outdir>/evaluation`.

- Run rsmsummarize on all of the fold directories and save output under
  `<outdir>/fold-summary`.

- Run rsmtool on the full training set to generate a final model along
  with a report including model/feature descriptives, all saved under
  `<outdir>/final-model`.

:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
"""

import logging
from os import listdir, makedirs
import sys
from os.path import abspath, exists, join
from pathlib import Path

from joblib.parallel import Parallel, delayed
from tqdm import tqdm

from .configuration_parser import Configuration, configure
from .rsmeval import run_evaluation
from .rsmtool import run_experiment
from .rsmsummarize import run_summary
from .utils.commandline import ConfigurationGenerator, setup_rsmcmd_parser
from .utils.constants import VALID_PARSER_SUBCOMMANDS
from .utils.cross_validation import combine_fold_prediction_files, create_xval_files, process_fold
from .utils.logging import LogFormatter, tqdm_joblib, get_file_logger
from .writer import DataWriter

# a constant defining all of the sections we can use when
# generating the final model report in the last stage
FINAL_MODEL_SECTION_LIST = ['feature_descriptives',
                            'features_by_group',
                            'preprocessed_features',
                            'dff_by_group',
                            'model',
                            'intermediate_file_paths',
                            'sysinfo']


def run_cross_validation(config_file_or_obj_or_dict, output_dir, silence_tqdm=False):
    """
    Run cross-validation experiment.

    Parameters
    ----------
    config_file_or_obj_or_dict : str or pathlib.Path or dict or Configuration
        Path to the experiment configuration file either as a string
        or as a ``pathlib.Path`` object. Users can also pass a
        ``Configuration`` object that is in memory or a Python dictionary
        with keys corresponding to fields in the configuration file. Given a
        configuration file, any relative paths in the configuration file
        will be interpreted relative to the location of the file. Given a
        ``Configuration`` object, relative paths will be interpreted
        relative to the ``configdir`` attribute, that _must_ be set. Given
        a dictionary, the reference path is set to the current directory.
    output_dir : str
        Path to the experiment output directory.
    silence_tqdm : bool, optional
        Whether to silence the progress bar that is shown when running
        rsmtool for each fold. This option should only be used when
        running the unit tests.
        Defaults to ``False``.

    Raises
    ------
    IOError
        If ``output_dir`` already contains the output of a previous experiment.

    """
    logger = logging.getLogger(__name__)

    # Get absolute paths to output directories
    foldsdir = abspath(join(output_dir, 'folds'))
    summarydir = abspath(join(output_dir, 'fold-summary'))
    evaldir = abspath(join(output_dir, 'evaluation'))
    modeldir = abspath(join(output_dir, 'final-model'))

    # Make directories, if necessary
    makedirs(foldsdir, exist_ok=True)
    makedirs(summarydir, exist_ok=True)
    makedirs(evaldir, exist_ok=True)
    makedirs(modeldir, exist_ok=True)

    # Raise an error if the specified output directory
    # already contains a non-empty `folds` directory;
    # we do not allow overwriting the output directory
    # for cross-validation experiments because of the
    # multitude of moving parts
    non_empty_foldsdir = exists(foldsdir) and listdir(foldsdir)
    if non_empty_foldsdir:
        raise IOError("'{}' already contains a non-empty 'folds' "
                      "directory.".format(output_dir))

    configuration = configure('rsmxval', config_file_or_obj_or_dict)

    logger.info("Saving configuration file.")
    with open(Path(output_dir) / "rsmxval.json", "w") as outfh:
        outfh.write(str(configuration))

    # create any cross-validation related files that are needed and
    # get a data frame containing the training set and the final
    # number of folds that are to be used in the experiment
    df_train, folds = create_xval_files(configuration, output_dir, logger=logger)

    # run RSMTool in parallel on each fold using joblib
    logger.info("Running RSMTool on each fold in parallel")
    with tqdm_joblib(tqdm(desc="Progress", total=folds, disable=silence_tqdm)):
        Parallel(n_jobs=folds)(delayed(process_fold)(fold_num, foldsdir)
                               for fold_num in range(1, folds + 1))

    # generate an rsmsummarize configuration file
    logger.info("Creating fold summary")
    given_file_format = configuration.get("file_format")
    fold_summary_configdict = {
        "summary_id": f"{configuration['experiment_id']}_fold_summary",
        "experiment_dirs": [f"{foldsdir}/{fold_num:02}"
                            for fold_num in range(1, folds + 1)],
        "description": f"{configuration['description']} (Fold Summary)",
        "file_format": given_file_format,
        "use_thumbnails": f"{configuration['use_thumbnails']}"
    }
    fold_summary_configuration = Configuration(fold_summary_configdict,
                                               configdir=summarydir,
                                               context="rsmsummarize")

    # run rsmsummarize on all of the fold directories
    summary_logger = get_file_logger("fold-summary",
                                     Path(summarydir) / "rsmsummarize.log")
    run_summary(fold_summary_configuration, summarydir, False, logger=summary_logger)

    # combine all of the fold prediction files for evaluation
    df_predictions = combine_fold_prediction_files(foldsdir, given_file_format)

    # if there were subgroups and a second human score column specified, then
    # we need to add those to the combined predictions file as well
    id_column = configuration["id_column"]
    columns_to_use = [id_column]
    if subgroups := configuration["subgroups"]:
        columns_to_use.extend(subgroups)
    if second_human_score_column := configuration["second_human_score_column"]:
        columns_to_use.append(second_human_score_column)
    if len(columns_to_use) > 1:
        df_to_add = df_train[columns_to_use]
        df_predictions = df_predictions.merge(df_to_add,
                                              left_on="spkitemid",
                                              right_on=id_column)
        # drop any extra ID column if we have added one
        if id_column != "spkitemid":
            df_predictions.drop(id_column, axis=1, inplace=True)

    # write out the predictions file to disk
    predictions_file_prefix = str(Path(evaldir) / "xval_predictions")
    DataWriter.write_frame_to_file(df_predictions,
                                   predictions_file_prefix,
                                   file_format=given_file_format,
                                   index=False)

    # create a new rsmeval configuration dictionary;
    # note that we do not want to set "id_column" and "human_score_column"
    # here since those columns already have default names given that the
    # predictions are generated from RSMTool
    evaluation_configdict = {
        "experiment_id": f"{configuration['experiment_id']}_evaluation",
        "description": f"{configuration['description']} (Evaluating Fold Predictions)",
        "predictions_file": f"{predictions_file_prefix}.{given_file_format}",
        "system_score_column": "scale" if configuration["use_scaled_predictions"] else "raw",
        "scale_with": "asis" if configuration["use_scaled_predictions"] else "raw",
        "trim_min": int(f"{configuration['trim_min']}"),
        "trim_max": int(f"{configuration['trim_max']}"),
        "file_format": given_file_format
    }

    # copy over the relevant configuration fields from the main configuration
    for field_name in ["second_human_score_column",
                       "candidate_column",
                       "flag_column",
                       "exclude_zero_scores",
                       "min_items_per_candidate",
                       "rater_error_variance",
                       "subgroups",
                       "trim_tolerance",
                       "use_thumbnails"]:
        evaluation_configdict[field_name] = configuration[field_name]

    # create an rsmeval configuration object with this dictionary
    evaluation_configuration = Configuration(evaluation_configdict,
                                             configdir=evaldir,
                                             context="rsmeval")

    # run rsmeval on the combined predictions file
    logger.info("Evaluating combined fold predictions")
    eval_logger = get_file_logger("evaluation", Path(evaldir) / "rsmeval.log")
    run_evaluation(evaluation_configuration, evaldir, False, logger=eval_logger)

    # run rsmtool on the full dataset and generate only model/feature
    # descriptives report; we will use the dummy test set that we
    # created by calling `create_xval_files()` for this purpose
    logger.info("Training model on full data")
    model_logger = get_file_logger("final_model", Path(modeldir) / "rsmtool.log")
    final_rsmtool_configdict = configuration.to_dict().copy()
    final_rsmtool_configdict["experiment_id"] = f"{configuration['experiment_id']}_model"
    final_rsmtool_configdict["test_file"] = str(Path(modeldir) / f"dummy_test.{given_file_format}")
    final_rsmtool_configdict["test_label_column"] = final_rsmtool_configdict["train_label_column"]

    # remove by-group sections if we don't have the info
    sections_to_use = []
    for section in FINAL_MODEL_SECTION_LIST:
        if section.endswith("_by_group") and not subgroups:
            continue
        sections_to_use.append(section)
    final_rsmtool_configdict["general_sections"] = sections_to_use
    final_rsmtool_configuration = Configuration(final_rsmtool_configdict,
                                                configdir=configuration.configdir,
                                                context="rsmtool")
    run_experiment(final_rsmtool_configuration, modeldir, False, logger=model_logger)


def main():  # noqa: D103

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

    logging.root.setLevel(logging.INFO)
    logger = logging.getLogger(__name__)

    # set up an argument parser via our helper function
    parser = setup_rsmcmd_parser('rsmxval',
                                 uses_output_directory=True,
                                 allows_overwriting=False,
                                 uses_subgroups=True)

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
        logger.info('Output directory: {}'.format(args.output_dir))
        run_cross_validation(abspath(args.config_file),
                             abspath(args.output_dir))

    else:

        # when generating, log to stderr
        logging.root.addHandler(stderr_handler)

        # auto-generate an example configuration and print it to STDOUT
        generator = ConfigurationGenerator('rsmxval',
                                           as_string=True,
                                           suppress_warnings=args.quiet,
                                           use_subgroups=args.subgroups)
        configuration = generator.interact() if args.interactive else generator.generate()
        print(configuration)


if __name__ == '__main__':
    main()
