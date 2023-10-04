"""
Various utility functions used for cross-validation.

:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""
import logging
from pathlib import Path
from shutil import copyfile

from pandas import concat
from sklearn.model_selection import KFold, LeaveOneGroupOut
from skll.config.utils import load_cv_folds

from rsmtool.reader import DataReader
from rsmtool.rsmtool import run_experiment
from rsmtool.utils.logging import get_file_logger
from rsmtool.writer import DataWriter


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
    df_train : pandas DataFrame
        DataFrame containing the raw training data file.
    folds : int
        The number of folds for which the files were created.

    Raises
    ------
    FileNotFoundError
        If the training data file specified in the configuration is not found.
    """
    # instantiate a logger if one is not given
    logger = logger if logger else logging.getLogger(__name__)

    # get the paths to the "<output_dir>/folds" directory as well as to
    # the "<output_dir>/final-model" directory since those are the two
    # directories under which we will write files
    foldsdir = Path(output_dir) / "folds"
    modeldir = Path(output_dir) / "final-model"

    # get the file format specified by the user
    given_file_format = configuration["file_format"]

    # locate the training data file, the folds file, and feature file(s)
    located_filepaths = {}
    located_filepaths["train"] = DataReader.locate_files(
        configuration.get("train_file"), configuration.configdir
    )
    if not located_filepaths["train"]:
        raise FileNotFoundError(
            f"The training data file was not found: " f"{repr(configuration.get('train_file'))}"
        )

    # "features" could be a list of features so check for that
    additional_filenames = ["folds_file", "feature_subset_file"]
    if isinstance(configuration.get("features"), str):
        additional_filenames.append("features")

    for additional_filename in additional_filenames:
        if additional_file := configuration.get(additional_filename):
            located_filepaths[additional_filename] = DataReader.locate_files(
                additional_file, configuration.configdir
            )

    # read the training file into a dataframe making sure that the specified
    # ID column is read as a string
    df_train = DataReader.read_from_file(
        located_filepaths["train"], converters={configuration["id_column"]: str}
    )

    # we need to sub-sample the full training data file to create a dummy
    # test file that we need to use when running RSMTool on the full
    # training dataset to get the model/feature descriptives; we use 10%
    # of the training data to create this dummy test set
    df_sampled_test = df_train.sample(frac=0.1, replace=False, random_state=1234567890, axis=0)
    DataWriter.write_frame_to_file(
        df_sampled_test,
        str(modeldir / "dummy_test"),
        file_format=given_file_format,
        index=False,
    )

    # read the folds file if it exists, otherwise use specified number of folds
    folds_file = located_filepaths.get("folds_file")
    if folds_file and Path(folds_file).exists():
        cv_folds = load_cv_folds(located_filepaths["folds_file"])
        folds = len(set(cv_folds.values()))
        logger.info(f"Using {folds} folds specified in {located_filepaths['folds_file']}")
        logo = LeaveOneGroupOut()
        id_column = configuration.get("id_column")
        try:
            train_ids = df_train[id_column]
        except KeyError:
            logger.error(
                "Column f{id_column} not found! Check the value of "
                "'id_column' in the configuration file."
            )
        else:
            fold_groups = [cv_folds[train_id] for train_id in train_ids.values]
            fold_generator = logo.split(range(len(df_train)), y=None, groups=fold_groups)
    else:
        # if we are in this code path but the configuration did
        # specify "folds_file", then we must not have found the file
        # so raise a warning to that effect
        if configuration.get("folds_file"):
            logging.warning("Specified folds file not found. Ignoring.")
        folds = configuration.get("folds")
        logger.info(f"Generating {folds} folds after shuffling")
        kfold = KFold(n_splits=folds, random_state=1234567890, shuffle=True)
        fold_generator = kfold.split(range(len(df_train)))

    # iterate over each of the folds and generate an rsmtool configuration file
    # which is then saved to disk in a directory specific to each fold
    for fold_num, (fold_train_indices, fold_test_indices) in enumerate(fold_generator, start=1):
        # get the train and test files for this fold and
        # write them out to disk
        df_fold_train = df_train.loc[fold_train_indices]
        df_fold_test = df_train.loc[fold_test_indices]
        this_fold_dir = Path(foldsdir) / f"{fold_num:02}"
        this_fold_dir.mkdir()
        fold_train_file_prefix = str(this_fold_dir / "train")
        DataWriter.write_frame_to_file(
            df_fold_train,
            fold_train_file_prefix,
            file_format=given_file_format,
            index=False,
        )
        fold_test_file_prefix = str(this_fold_dir / "test")
        DataWriter.write_frame_to_file(
            df_fold_test,
            fold_test_file_prefix,
            file_format=given_file_format,
            index=False,
        )

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
        fold_configuration["description"] = f"{configuration['description']} (Fold {fold_num:02})"

        # if "feature_subset" was specified in the original configuration,
        # then we need it in the fold configuration as well
        if subset := configuration.get("feature_subset"):
            fold_configuration["feature_subset"] = subset

        # update "configdir" and "context" attributes in configuration file
        fold_configuration.configdir = str(this_fold_dir)
        fold_configuration.context = "rsmtool"
        with open(this_fold_dir / "rsmtool.json", "w") as fold_config_file:
            fold_config_file.write(str(fold_configuration))

        # copy "features" or "features_subset_file" to the
        # same directory as where the configuration was saved;
        # note that if features is a list instead of a file,
        # it will get ignored automatically
        for filename in ["features", "feature_subset_file"]:
            if filepath := located_filepaths.get(filename):
                copyfile(filepath, this_fold_dir / Path(filepath).name)

    return df_train, folds


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


def combine_fold_prediction_files(foldsdir, file_format="csv"):
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
        df_fold_predictions = DataReader.read_from_file(
            prediction_file, converters={"spkitemid": str}
        )
        prediction_dfs.append(df_fold_predictions)

    # concatenate all of the predictions into a single frame using
    # "spkitemid" column which must exist since this is the output of RSMTool
    df_all_predictions = concat(prediction_dfs).reset_index(drop=True)

    return df_all_predictions
