#!/usr/bin/env python

"""
Run a cross-validation experiment using RSMTool, RSMEval, and RSMSummarize.

The complete cross-validation workflow is as follows:

- Divide the training data file into folds either by randomly shuffling
  and then using a k-fold split or by using a pre-specified folds file
  that provides a fold number for each example ID in the data

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
from shutil import copyfile
import sys
from os.path import abspath, exists, join
from pathlib import Path

from joblib.parallel import Parallel, delayed
from pandas import concat
from skll.config.utils import load_cv_folds
from sklearn.model_selection import KFold, LeaveOneGroupOut
from tqdm import tqdm

from .configuration_parser import Configuration, configure
from .reader import DataReader
from .rsmeval import run_evaluation
from .rsmtool import run_experiment
from .rsmsummarize import run_summary
from .utils.commandline import ConfigurationGenerator, setup_rsmcmd_parser
from .utils.constants import VALID_PARSER_SUBCOMMANDS
from .utils.logging import LogFormatter, tqdm_joblib, get_file_logger
from .writer import DataWriter


def create_xval_files(configuration, output_dir, logger=None):
    """
    Create all files needed for the cross-validation experiment.
    
    Parameters
    ----------
    configuration : rsmtool.configuration_parser.Configuration
        A Configuration object holding the user-specified configuration
        for "rsmxval".
    output_dir : str
        Path to the output directory specified for "rsmxval".
    logger : None, optional
        Logger object used to log messages from this function.
    
    Returns
    -------
    num_folds : int
        The number of folds for which the files were created.
    
    Raises
    ------
    FileNotFoundError
        If the training data file specified in the configuration is not found.
    """

    # get the paths to the "<output_dir>/folds" directory as well as to
    # the "<output_dir>/final-model" directory since those are the two 
    # directories under which we will write files
    foldsdir = Path(output_dir) / "folds"
    modeldir = Path(output_dir) / "final-model"

    # get the file format specified by the user
    given_file_format = configuration["file_format"]

    # locate the training data file, the folds file, and feature file(s)
    located_filepaths = {}
    located_filepaths["train"] = DataReader.locate_files(configuration.get("train_file"), 
                                                         configuration.configdir)
    if not located_filepaths["train"]:
        raise FileNotFoundError('The training data file was not found: '
                                '{}'.format(repr(configuration.get("train_file"))))

    for additional_filename in ["folds_file", "features", "feature_subset_file"]:
        if additional_file := configuration.get(additional_filename):
            located_filepaths[additional_filename] = DataReader.locate_files(additional_file, 
                                                                             configuration.configdir)

    # read the training file into a dataframe
    df_train = DataReader.read_from_file(located_filepaths["train"])

    # we need to sub-sample the full training data file to create a dummy
    # test file that we need to use when running RSMTool on the full 
    # training dataset to get the model/feature descriptives; we use 10%
    # of the training data to create this dummy test set
    df_sampled_test = df_train.sample(frac=0.1, 
                                      replace=False,
                                      random_state=1234567890,
                                      axis=0)
    DataWriter.write_frame_to_file(df_sampled_test, 
                                   str(modeldir / "dummy_test"),
                                   file_format=given_file_format,
                                   index=False)

    # read the folds file if it exists, otherwise use specified number of folds
    if "folds_file" in located_filepaths and Path(located_filepaths["folds_file"]).exists():
        cv_folds = load_cv_folds(located_filepaths["folds_file"])
        num_folds = len(set(cv_folds.values()))
        logger.info(f"Using {num_folds} folds specified in {located_filepaths['folds_file']}")
        logo = LeaveOneGroupOut()
        id_column = configuration.get("id_column")
        try:
            train_ids = df_train[id_column]
        except KeyError:
            logger.error("Column f{id_column} not found! Check the value of "
                         "'id_column' in the configuration file.")
            sys.exit(1)
        else:
            fold_groups = [cv_folds[train_id] for train_id in train_ids.values]
            fold_generator = logo.split(range(len(df_train)), y=None, groups=fold_groups)
    else:
        num_folds = configuration.get("num_folds")
        logger.info(f"Generating {num_folds} folds after shuffling")
        kfold = KFold(n_splits=num_folds, random_state=1234567890, shuffle=True)
        fold_generator = kfold.split(range(len(df_train)))

    # iterate over each of the folds and generate an rsmtool configuration file
    # which is then saved to disk in a directory specific to each fold
    for fold_num, (fold_train_indices, 
                   fold_test_indices) in enumerate(fold_generator, start=1):

        # get the train and test files for this fold and
        # write them out to disk
        df_fold_train = df_train.loc[fold_train_indices]
        df_fold_test = df_train.loc[fold_test_indices]
        this_fold_dir = Path(foldsdir) / f"{fold_num:02}"
        this_fold_dir.mkdir()
        fold_train_file_prefix = str(this_fold_dir / "train")
        DataWriter.write_frame_to_file(df_fold_train,
                                       fold_train_file_prefix,
                                       file_format=given_file_format,
                                       index=False)
        fold_test_file_prefix = str(this_fold_dir / "test.csv")
        DataWriter.write_frame_to_file(df_fold_test,
                                       fold_test_file_prefix,
                                       file_format=given_file_format,
                                       index=False)

        # update the value of "train_file" and add "test_file"
        # & "test_label_column" in the configuration file
        fold_configuration = configuration.copy(deep=True)
        fold_configuration["train_file"] = f"{fold_train_file_prefix}.{given_file_format}"
        fold_configuration["test_file"] = f"{fold_test_file_prefix}.{given_file_format}"
        fold_configuration["test_label_column"] = configuration["train_label_column"]

        # use the same file format as the user specified
        fold_configuration["file_format"] = given_file_format

        # update "experiment_id" and "description" in configuration file
        fold_configuration["experiment_id"] = f"{configuration['experiment_id']}_fold{fold_num:02}"
        fold_configuration["description"] = f"{configuration['description']} (Fold {fold_num:02})",

        # update "configdir" and "context" attributes in configuration file
        fold_configuration.configdir = str(this_fold_dir)
        fold_configuration.context = "rsmtool"
        with open(this_fold_dir / "rsmtool.json", "w") as fold_config_file:
            fold_config_file.write(str(fold_configuration))

        # copy "features" or "features_subset_file" to the
        # same directory as where the configuration was saved
        for filename in ["features", "feature_subset_file"]:
            if filename in located_filepaths:
                filepath = located_filepaths[filename]
                copyfile(filepath, this_fold_dir / Path(filepath).name)

    return num_folds


def process_fold(fold_num, foldsdir):
    """
    Run RSMTool on the specified numbered fold.
    
    Parameters
    ----------
    fold_num : int
        The number of the fold to run RSMTool on.
    foldsdir : str
        The directory which stores the output for each fold experiment.
    """
    # create a directory name specific to the given fold
    fold_dir = Path(foldsdir) / f"{fold_num:02}"

    # get a file logger since we cannot show the log for RSMTool on screen
    log_file_path = fold_dir / "rsmtool.log"
    logger = get_file_logger(f"fold{fold_num:02}", log_file_path)

    # run RSMTool on the given fold and have it log its output to the file
    config_file = fold_dir / "rsmtool.json"
    run_experiment(config_file, fold_dir, False, logger=logger)


def combine_fold_prediction_files(foldsdir, file_format):
    """
    Combine predictions from all folds into a single data frame.

    Parameters
    ----------
    foldsdir : str
        The directory which stores the output for each fold experiment.
    file_format : str
        The file format (extension) for the file to be written to disk.
        One of {"csv", "xlsx", "tsv"}.
        Defaults to "csv".        
    
    Returns
    -------
    df_all_predictions : pandas DataFrame
        Data frame containing the combined predictions for all folds.
    """
    # initialize empty list to hold each prediction data frame
    prediction_dfs = []

    # iterate over each fold in the given directory & read and save predictions
    for prediction_file in Path(foldsdir).glob(f"**/*pred_processed*.{file_format}"):
        df_fold_predictions = DataReader.read_from_file(prediction_file, converters={'spkitemid': str})
        prediction_dfs.append(df_fold_predictions)

    # concatenate all of the predictions into a single frame using 
    # "spkitemid" column which must exist since this is the output of RSMTool
    df_all_predictions = concat(prediction_dfs, keys="spkitemid").reset_index(drop=True)

    return df_all_predictions


def run_cross_validation(config_file_or_obj_or_dict,
                         output_dir):
    """
    Run cross-validation experiment.
    
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
    output_dir : str
        Path to the experiment output directory.
    
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

    # check if an explicit folds file is provided
    num_folds = create_xval_files(configuration, output_dir, logger=logger)
    
    # run RSMTool in parallel on each fold using joblib
    logger.info("Running RSMTool on each fold in parallel")
    with tqdm_joblib(tqdm(desc="Progress", total=num_folds)):
        Parallel(n_jobs=num_folds)(delayed(process_fold)(fold_num, foldsdir) for fold_num in range(1, num_folds + 1))

    # generate an rsmsummarize configuration file
    logger.info("Creating fold summary")
    given_file_format = configuration.get("file_format")
    fold_summary_configdict = {
        "summary_id": f"{configuration['experiment_id']}_fold_summary",
        "experiment_dirs": [f"{foldsdir}/{fold_num:02}" for fold_num in range(1, num_folds + 1)],
        "description": f"{configuration['description']} (Fold Summary)",
        "file_format": given_file_format,
        "use_thumbnails": f"{configuration['use_thumbnails']}"
    }
    fold_summary_configuration = Configuration(fold_summary_configdict, configdir=summarydir, context="rsmsummarize")

    # run rsmsummarize on all of the fold directories
    summary_logger = get_file_logger("fold-summary", Path(summarydir) / "rsmsummarize.log")
    run_summary(fold_summary_configuration, summarydir, False, logger=summary_logger)

    # combine all of the fold prediction files for evaluation
    df_predictions = combine_fold_prediction_files(foldsdir, given_file_format)
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
    evaluation_configuration = Configuration(evaluation_configdict, configdir=evaldir, context="rsmeval")

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
    final_rsmtool_configdict["test_file"] = str(Path(modeldir) / f"dummy_test.{given_file_format}")
    final_rsmtool_configdict["test_label_column"] = final_rsmtool_configdict["train_label_column"]
    final_rsmtool_configdict["general_sections"] = [
        "feature_descriptives",
        "preprocessed_features",
        "model",
        "intermediate_file_paths",
        "sysinfo"
    ]
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
        generator = ConfigurationGenerator('rsmtool',
                                           as_string=True,
                                           suppress_warnings=args.quiet,
                                           use_subgroups=args.subgroups)
        configuration = generator.interact() if args.interactive else generator.generate()
        print(configuration)


if __name__ == '__main__':
    main()
