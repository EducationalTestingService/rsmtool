#!/usr/bin/env python
"""
Generate predictions on new data from rsmtool models.

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

import glob
import logging
import os
import sys
from os.path import abspath, basename, dirname, exists, join, normpath, split, splitext

import numpy as np
import pandas as pd

from .configuration_parser import configure
from .modeler import Modeler
from .preprocessor import FeaturePreprocessor
from .reader import DataReader
from .utils.commandline import CmdOption, ConfigurationGenerator, setup_rsmcmd_parser
from .utils.constants import VALID_PARSER_SUBCOMMANDS
from .utils.logging import LogFormatter
from .writer import DataWriter


def fast_predict(
    input_features,
    modeler,
    df_feature_info,
    trim_min=None,
    trim_max=None,
    trim_tolerance=0.4998,
    train_predictions_mean=None,
    train_predictions_sd=None,
    h1_mean=None,
    h1_sd=None,
    logger=None,
):
    """
    Compute predictions for a single instance against given model.

    The main difference between this function and the ``compute_and_save_predictions()``
    function is that the former is meant for batch prediction and reads
    all its inputs from disk and writes its outputs to disk. This function,
    however, is meant for real-time inference rather than batch. To this end,
    it operates *entirely* in memory. Note that there is still a bit of overlap
    between the two computation paths since we want to use the RSMTool API
    as much as possible.

    This function should only be used when the goal is to generate predictions
    using RSMTool models in production. The user should read everything from disk
    in a separate thread/function and pass the inputs to this function.

    Note that this function only computes regular predictions, not expected
    scores.

    Parameters
    ----------
    input_features : dict[str, float]
        A dictionary containing the features for the instance for which to
        generate the model predictions. The keys should be names of the
        features on which the model was trained and the values should
        be the *raw* feature values.
    modeler : rsmtool.modeler.Modeler object
        The RSMTool modeler object using which the predictions
        are to be generated. This object should be created from the already
        existing model file in the "output" directory of the previously run
        RSMTool experiment.
    df_feature_info : pandas DataFrame
        A DataFrame containing the information regarding the model features.
        The index of the dataframe should be the names of the features and
        the columns should be:

        - "sign" : 1 or -1.  Indicates whether the feature value needs to
          be multiplied by -1.
        - "transform" : :ref:`transformation <select_transformations_rsmtool>`
          that needs to be applied to this feature.
        - "train_mean", "train_sd" : mean and standard deviation for outlier
          truncation.
        - "train_transformed_mean", "train_transformed_sd" : mean and standard
          deviation for computing z-scores.

        This dataframe should be read from the "feature.csv" file under the
        "output" directory of the previously run RSMTool experiment.
    trim_min : int, optional
        The lowest possible integer score that the machine should predict.
        If ``None``, raw predictions will not be trimmed.
        Defaults to ``None``.
    trim_max : int, optional
        The highest possible integer score that the machine should predict.
        If ``None``, raw predictions will not be trimmed.
        Defaults to ``None``.
    trim_tolerance : float, optional
       The single numeric value that will be used to pad the trimming range
       specified in ``trim_min`` and ``trim_max``.
       Defaults to 0.4998.
    train_predictions_mean : float, optional
       The mean of the predictions on the training set used to re-scale the
       predictions. May be read from the "postprocessing_params.csv" file
       under the "output" directory of the RSMTool experiment used to train
       the model. If ``None``, predictions will not be scaled.
       Defaults to ``None``.
    train_predictions_sd : float, optional
       The standard deviation of the predictions on the training set used to
       re-scale the predictions. May be read from the "postprocessing_params.csv"
       file under the "output" directory of the RSMTool experiment used to train
       the model.If ``None``, predictions will not be scaled.
       Defaults to ``None``.
    h1_mean : float, optional
       The mean of the human scores in the training set also used to re-scale
       the predictions. May be read from the "postprocessing_params.csv" file
       under the "output" directory of the RSMTool experiment used to train
       the model. If ``None``, predictions will not be scaled.
       Defaults to ``None``.
    h1_sd : float, optional
       The standard deviation of the human scores in the training set used to
       re-scale the predictions. May be read from the "postprocessing_params.csv"
       file under the "output" directory of the RSMTool experiment used to train
       the model. If ``None``, predictions will not be scaled.
       Defaults to ``None``.
    logger : logging object, optional
        A logging object. If ``None`` is passed, get logger from ``__name__``.
        Defaults to ``None``.

    Returns
    -------
    dict[str, float]
        A dictionary containing the raw, scaled, trimmed, and rounded
        predictions for the input features. It always contains the
        "raw" key and may contain the following additional keys depending
        on the availability of the various optional arguments:
        "raw_trim", "raw_trim_round", "scale", "scale_trim",
        and "scale_trim_round".

    Raises
    ------
    ValueError
        If ``input_features`` contains any non-numeric features.
    """
    # initialize a logger if none provided
    logger = logger if logger else logging.getLogger(__name__)

    # instantiate a feature preprocessor
    preprocessor = FeaturePreprocessor(logger=logger)

    # convert the given features to a data frame and add the "spkitemid" column
    df_input_features = pd.DataFrame([input_features])
    df_input_features["spkitemid"] = "RESPONSE"

    # preprocess the input features so that they match what the model expects
    try:
        df_processed_features, _ = preprocessor.preprocess_new_data(
            df_input_features, df_feature_info
        )
    except ValueError:
        raise ValueError("Input features must not contain non-numeric values.") from None

    # now compute the raw prediction for the given features
    df_predictions = modeler.predict(df_processed_features)

    # compute scaled prediction if the scaling parameters are available
    if train_predictions_mean and train_predictions_sd and h1_mean and h1_sd:
        df_predictions["scale"] = (
            (df_predictions["raw"] - train_predictions_mean) / train_predictions_sd
        ) * h1_sd + h1_mean

    # drop the spkitemid column since it's not needed from this point onwards
    df_predictions.drop("spkitemid", axis="columns", inplace=True)

    # trim both raw and scale predictions if ``trim_min`` and ``trim_max`` were specified
    if trim_min and trim_max:
        for column in df_predictions.columns:
            df_predictions[f"{column}_trim"] = preprocessor.trim(
                df_predictions[column], trim_min, trim_max, trim_tolerance
            )
            df_predictions[f"{column}_trim_round"] = np.rint(
                df_predictions[f"{column}_trim"]
            ).astype("int64")

    # return the predictions as a dictionary
    return df_predictions.to_dict(orient="records")[0]


def compute_and_save_predictions(
    config_file_or_obj_or_dict, output_file, feats_file=None, logger=None
):
    """
    Run rsmpredict using the given configuration.

    Generate predictions using given configuration file, object, or
    dictionary. Predictions are saved in ``output_file``. Optionally,
    pre-processed feature values are saved in ``feats_file``,
    if specified.

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
    logger : logging object, optional
        A logging object. If ``None`` is passed, get logger from ``__name__``.
        Defaults to ``None``.

    Raises
    ------
    FileNotFoundError
        If any of the files contained in ``config_file_or_obj_or_dict``
        cannot be located.
    FileNotFoundError
        If ``experiment_dir`` does not exist.
    FileNotFoundError
        If ``experiment_dir`` does not contain the required output
        needed from an rsmtool experiment.
    RuntimeError
        If the name of the output file does not end in
        ".csv", ".tsv", or ".xlsx".
    """
    logger = logger if logger else logging.getLogger(__name__)

    configuration = configure("rsmpredict", config_file_or_obj_or_dict)

    # get the experiment ID
    experiment_id = configuration["experiment_id"]

    # Get output format
    file_format = configuration.get("file_format", "csv")

    # Get DataWriter object
    writer = DataWriter(experiment_id)

    # get the input file containing the feature values
    # for which we want to generate the predictions
    input_features_file = DataReader.locate_files(
        configuration["input_features_file"], configuration.configdir
    )
    if not input_features_file:
        raise FileNotFoundError(f"Input file {configuration['input_features_file']} does not exist")

    experiment_dir = DataReader.locate_files(
        configuration["experiment_dir"], configuration.configdir
    )
    if not experiment_dir:
        raise FileNotFoundError(f"The directory {configuration['experiment_dir']} does not exist.")
    else:
        experiment_output_dir = normpath(join(experiment_dir, "output"))
        if not exists(experiment_output_dir):
            raise FileNotFoundError(
                f"The directory {experiment_dir} does not contain "
                f"the output of an rsmtool experiment."
            )

    # find all the .model files in the experiment output directory
    model_files = glob.glob(join(experiment_output_dir, "*.model"))
    if not model_files:
        raise FileNotFoundError(
            f"The directory {experiment_output_dir} does not " f"contain any rsmtool models."
        )

    experiment_ids = [splitext(basename(mf))[0] for mf in model_files]
    if experiment_id not in experiment_ids:
        raise FileNotFoundError(
            f"{experiment_output_dir} does not contain a model "
            f'for the experiment "{experiment_id}". The following '
            f"experiments are contained in this directory: {experiment_ids}"
        )

    # check that the directory contains outher required files
    required_file_types = ["feature", "postprocessing_params"]
    for file_type in required_file_types:
        expected_file_name = f"{experiment_id}_{file_type}.csv"
        if not exists(join(experiment_output_dir, expected_file_name)):
            raise FileNotFoundError(
                f"{experiment_output_dir} does not contain the "
                f"required file {expected_file_name} that was "
                f"generated during the original model training."
            )

    logger.info("Reading input files.")

    feature_info = join(experiment_output_dir, f"{experiment_id}_feature.csv")

    post_processing = join(experiment_output_dir, f"{experiment_id}_postprocessing_params.csv")

    file_paths = [input_features_file, feature_info, post_processing]
    file_names = ["input_features", "feature_info", "postprocessing_params"]

    converters = {"input_features": configuration.get_default_converter()}

    # Initialize the reader
    reader = DataReader(file_paths, file_names, converters)
    data_container = reader.read(kwargs_dict={"feature_info": {"index_col": 0}})

    # load the Modeler to generate the predictions
    model = Modeler.load_from_file(join(experiment_output_dir, f"{experiment_id}.model"))

    # Add the model to the configuration object
    configuration["model"] = model

    # Initialize the processor
    processor = FeaturePreprocessor(logger=logger)

    (processed_config, processed_container) = processor.process_data(
        configuration, data_container, context="rsmpredict"
    )

    # save the pre-processed features to disk if we were asked to
    if feats_file is not None:
        logger.info(f"Saving pre-processed feature values to {feats_file}")

        feats_dir = dirname(feats_file)

        # create any directories needed for the output file
        os.makedirs(feats_dir, exist_ok=True)

        _, feats_filename = split(feats_file)
        feats_filename, _ = splitext(feats_filename)

        # Write out files
        writer.write_experiment_output(
            feats_dir,
            processed_container,
            include_experiment_id=False,
            dataframe_names=["features_processed"],
            new_names_dict={"features_processed": feats_filename},
            file_format=file_format,
        )

    if output_file.lower().endswith(".csv") or output_file.lower().endswith(".xlsx"):

        output_dir = dirname(output_file)
        _, filename = split(output_file)
        filename, _ = splitext(filename)

    else:
        output_dir = output_file
        filename = "predictions_with_metadata"

    # create any directories needed for the output file
    os.makedirs(output_dir, exist_ok=True)

    # save the predictions to disk
    logger.info("Saving predictions.")

    # Write out files
    writer.write_experiment_output(
        output_dir,
        processed_container,
        include_experiment_id=False,
        dataframe_names=["predictions_with_metadata"],
        new_names_dict={"predictions_with_metadata": filename},
        file_format=file_format,
    )

    # save excluded responses to disk
    if not processed_container.excluded.empty:

        # save the predictions to disk
        logger.info(
            f"Saving excluded responses to "
            f"{join(output_dir, '{}_excluded_responses.csv'.format(filename))}"
        )

        # Write out files
        writer.write_experiment_output(
            output_dir,
            processed_container,
            include_experiment_id=False,
            dataframe_names=["excluded"],
            new_names_dict={"excluded": f"{filename}_excluded_responses"},
            file_format=file_format,
        )


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

    # to set up the argument parser, we first need to instantiate options
    # specific to rsmpredict so we use the `CmdOption` namedtuples
    non_standard_options = [
        CmdOption(dest="output_file", help="output file where predictions will be saved."),
        CmdOption(
            dest="preproc_feats_file",
            help="if specified, the preprocessed features " "will be saved in this file",
            longname="features",
            required=False,
        ),
    ]

    # now call the helper function to instantiate the parser for us
    parser = setup_rsmcmd_parser(
        "rsmpredict",
        uses_output_directory=False,
        extra_run_options=non_standard_options,
    )

    # if we have no arguments at all then just show the help message
    if len(sys.argv) < 2:
        sys.argv.append("-h")

    # if the first argument is not one of the valid sub-commands
    # or one of the valid optional arguments, then assume that they
    # are arguments for the "run" sub-command. This allows the
    # old style command-line invocations to work without modification.
    if sys.argv[1] not in VALID_PARSER_SUBCOMMANDS + [
        "-h",
        "--help",
        "-V",
        "--version",
    ]:
        args_to_pass = ["run"] + sys.argv[1:]
    else:
        args_to_pass = sys.argv[1:]
    args = parser.parse_args(args=args_to_pass)

    # call the appropriate function based on which sub-command was run
    if args.subcommand == "run":

        # when running, log to stdout
        logging.root.addHandler(stdout_handler)

        # run the experiment
        preproc_feats_file = None
        if args.preproc_feats_file:
            preproc_feats_file = abspath(args.preproc_feats_file)
        compute_and_save_predictions(
            abspath(args.config_file),
            abspath(args.output_file),
            feats_file=preproc_feats_file,
        )

    else:

        # when generating, log to stderr
        logging.root.addHandler(stderr_handler)

        # auto-generate an example configuration and print it to STDOUT
        generator = ConfigurationGenerator(
            "rsmpredict", as_string=True, suppress_warnings=args.quiet
        )
        configuration = generator.interact() if args.interactive else generator.generate()
        print(configuration)


if __name__ == "__main__":
    main()
