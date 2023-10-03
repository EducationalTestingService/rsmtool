#!/usr/bin/env python

"""
Run an rsmtool experiment.

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

import logging
import sys
from os import listdir, makedirs
from os.path import abspath, exists, join

from .analyzer import Analyzer
from .configuration_parser import configure
from .modeler import Modeler
from .preprocessor import FeaturePreprocessor
from .reader import DataReader
from .reporter import Reporter
from .utils.commandline import ConfigurationGenerator, setup_rsmcmd_parser
from .utils.constants import VALID_PARSER_SUBCOMMANDS
from .utils.logging import LogFormatter
from .utils.wandb import init_wandb_run, log_configuration_to_wandb
from .writer import DataWriter


def run_experiment(
    config_file_or_obj_or_dict, output_dir, overwrite_output=False, logger=None, wandb_run=None
):
    """
    Run an rsmtool experiment using the given configuration.

    Run rsmtool experiment using the given configuration file, object, or
    dictionary. All outputs are generated under ``output_dir``. If
    ``overwrite_output`` is ``True``, any existing output in ``output_dir``
    is overwritten.

    Parameters
    ----------
    config_file_or_obj_or_dict : str or pathlib.Path or dict or Configuration
        Path to the experiment configuration file either a string
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
    overwrite_output : bool, optional
        If ``True``, overwrite any existing output under ``output_dir``.
        Defaults to ``False``.
    logger : logging object, optional
        A logging object. If ``None`` is passed, get logger from ``__name__``.
        Defaults to ``None``.
    wandb_run : wandb.Run
        A wandb run object that will be used to log artifacts and tables.
        If ``None`` is passed, a new wandb run will be initialized if
        wandb is enabled in the configuration. Defaults to ``None``.

    Raises
    ------
    FileNotFoundError
        If any of the files contained in ``config_file_or_obj_or_dict`` cannot
        be located.
    IOError
        If ``output_dir`` already contains the output of a previous experiment
        and ``overwrite_output`` is ``False``.
    ValueError
        If the current configuration specifies a non-linear model but
        ``output_dir`` already contains the output of a previous
        experiment that used a linear model with the same experiment ID.
    """
    logger = logger if logger else logging.getLogger(__name__)

    # create the 'output' and the 'figure' sub-directories
    # where all the experiment output such as the CSV files
    # and the box plots will be saved

    # Get absolute paths to output directories
    csvdir = abspath(join(output_dir, "output"))
    figdir = abspath(join(output_dir, "figure"))
    reportdir = abspath(join(output_dir, "report"))
    featuredir = abspath(join(output_dir, "feature"))

    # Make directories, if necessary
    makedirs(csvdir, exist_ok=True)
    makedirs(figdir, exist_ok=True)
    makedirs(reportdir, exist_ok=True)

    # Raise an error if the specified output directory
    # already contains a non-empty `output` directory, unless
    # `overwrite_output` was specified, in which case we assume
    # that the user knows what she is doing and simply
    # output a warning saying that the report might
    # not be correct.
    non_empty_csvdir = exists(csvdir) and listdir(csvdir)
    if non_empty_csvdir:
        if not overwrite_output:
            raise IOError(f"'{output_dir}' already contains a non-empty 'output' directory.")
        else:
            logger.warning(
                f"{output_dir} already contains a non-empty 'output' directory. "
                f"The generated report might contain unexpected information from "
                f"a previous experiment."
            )

    configuration = configure("rsmtool", config_file_or_obj_or_dict)

    logger.info("Saving configuration file.")
    configuration.save(output_dir)

    # If wandb logging is enabled, and wandb_run is not provided,
    # start a wandb run and log configuration
    if wandb_run is None:
        wandb_run = init_wandb_run(configuration)
    log_configuration_to_wandb(wandb_run, configuration)

    # Get output format
    file_format = configuration.get("file_format", "csv")

    # Get DataWriter object
    writer = DataWriter(configuration["experiment_id"], configuration.context, wandb_run)

    # Get the paths and names for the DataReader

    (file_names, file_paths_org) = configuration.get_names_and_paths(
        ["train_file", "test_file", "features", "feature_subset_file"],
        ["train", "test", "feature_specs", "feature_subset_specs"],
    )
    file_paths = DataReader.locate_files(file_paths_org, configuration.configdir)

    # if there are any missing files after trying to locate
    # all expected files, raise an error
    if None in file_paths:
        missing_file_paths = [
            file_paths_org[idx] for idx, path in enumerate(file_paths) if path is None
        ]
        raise FileNotFoundError(f"The following files were not found: {repr(missing_file_paths)}")

    # Use the default converter for both train and test
    converters = {
        "train": configuration.get_default_converter(),
        "test": configuration.get_default_converter(),
    }

    logger.info("Reading in all data from files.")

    # Initialize the reader
    reader = DataReader(file_paths, file_names, converters)
    data_container = reader.read()

    logger.info("Preprocessing all features.")

    # Initialize the processor
    processor = FeaturePreprocessor(logger=logger)

    (processed_config, processed_container) = processor.process_data(configuration, data_container)

    # Rename certain frames with more descriptive names
    # for writing out experiment files
    rename_dict = {
        "train_excluded": "train_excluded_responses",
        "test_excluded": "test_excluded_responses",
        "train_length": "train_response_lengths",
        "train_flagged": "train_responses_with_excluded_flags",
        "test_flagged": "test_responses_with_excluded_flags",
    }

    logger.info("Saving training and test set data to disk.")

    # Write out files
    writer.write_experiment_output(
        csvdir,
        processed_container,
        [
            "train_features",
            "test_features",
            "train_metadata",
            "test_metadata",
            "train_other_columns",
            "test_other_columns",
            "train_preprocessed_features",
            "test_preprocessed_features",
            "train_excluded",
            "test_excluded",
            "train_length",
            "test_human_scores",
            "train_flagged",
            "test_flagged",
        ],
        rename_dict,
        file_format=file_format,
    )

    # Initialize the analyzer
    analyzer = Analyzer(logger=logger)

    (_, analyzed_container) = analyzer.run_data_composition_analyses_for_rsmtool(
        processed_container, processed_config
    )

    # Write out files
    writer.write_experiment_output(csvdir, analyzed_container, file_format=file_format)

    logger.info(f"Training {processed_config['model_name']} model.")

    # Initialize modeler
    modeler = Modeler(logger=logger)

    modeler.train(processed_config, processed_container, csvdir, figdir, file_format)

    # Identify the features used by the model
    selected_features = modeler.get_feature_names()

    # Add selected features to processed configuration
    processed_config["selected_features"] = selected_features

    # Write out files
    writer.write_feature_csv(
        featuredir, processed_container, selected_features, file_format=file_format
    )

    features_data_container = processed_container.copy()

    # Get selected feature info, and write out to file
    df_feature_info = features_data_container.feature_info.copy()
    df_selected_feature_info = df_feature_info[df_feature_info["feature"].isin(selected_features)]
    selected_feature_dataset_dict = {
        "name": "selected_feature_info",
        "frame": df_selected_feature_info,
    }

    features_data_container.add_dataset(selected_feature_dataset_dict, update=True)

    writer.write_experiment_output(
        csvdir,
        features_data_container,
        dataframe_names=["selected_feature_info"],
        new_names_dict={"selected_feature_info": "feature"},
        file_format=file_format,
    )

    logger.info("Running analyses on training set.")

    (_, train_analyzed_container) = analyzer.run_training_analyses(
        processed_container, processed_config
    )

    # Write out files
    writer.write_experiment_output(
        csvdir, train_analyzed_container, reset_index=True, file_format=file_format
    )

    # Use only selected features for predictions
    columns_for_prediction = ["spkitemid", "sc1"] + selected_features
    train_for_prediction = processed_container.train_preprocessed_features[columns_for_prediction]
    test_for_prediction = processed_container.test_preprocessed_features[columns_for_prediction]

    logged_str = "Generating training and test set predictions"
    logged_str += " (expected scores)." if configuration["predict_expected_scores"] else "."
    logger.info(logged_str)
    (pred_config, pred_data_container) = modeler.predict_train_and_test(
        train_for_prediction, test_for_prediction, processed_config
    )

    # Save modeler instance
    modeler.feature_info = processed_container.feature_info.copy()
    modeler.feature_info.set_index("feature", inplace=True)
    (
        modeler.trim_min,
        modeler.trim_max,
        modeler.trim_tolerance,
    ) = configuration.get_trim_min_max_tolerance()
    pred_config_dict = pred_config.to_dict()
    for key, attr_name in [
        ("train_predictions_mean", "train_predictions_mean"),
        ("train_predictions_sd", "train_predictions_sd"),
        ("human_labels_mean", "h1_mean"),
        ("human_labels_sd", "h1_sd"),
    ]:
        setattr(modeler, attr_name, pred_config_dict[key])
    logger.info("Saving model.")
    modeler.save(join(csvdir, f"{configuration['experiment_id']}.model"))

    # Write out files
    writer.write_experiment_output(
        csvdir,
        pred_data_container,
        new_names_dict={"pred_test": "pred_processed"},
        file_format=file_format,
    )

    original_coef_file = join(csvdir, f"{pred_config['experiment_id']}_coefficients.{file_format}")

    # If coefficients file exists, then try to generate the scaled
    # coefficients and save them to a file
    if exists(original_coef_file):
        logger.info("Scaling the coefficients and saving them to disk")
        try:
            # scale coefficients, and return DataContainer w/ scaled coefficients
            scaled_data_container = modeler.scale_coefficients(pred_config)

        # raise an error if the coefficient file exists but the
        # coefficients are not available for the current model
        # which can happen if the user is re-running the same experiment
        # with the same ID but with a non-linear model whereas the previous
        # run of the same ID was with a linear model and the user has not
        # cleared the directory
        except RuntimeError:
            raise ValueError(
                "It appears you previously ran an experiment with the "
                "same ID using a linear model and saved its output to "
                "the same directory. That output is interfering with "
                "the current experiment. Either clear the contents "
                "of the output directory or re-run the current "
                "experiment using a different experiment ID."
            )
        else:
            # Write out scaled coefficients to disk
            writer.write_experiment_output(csvdir, scaled_data_container, file_format=file_format)

    # Add processed data_container frames to pred_data_container
    new_pred_data_container = pred_data_container + processed_container

    logger.info("Running prediction analyses.")
    (
        pred_analysis_config,
        pred_analysis_data_container,
    ) = analyzer.run_prediction_analyses(new_pred_data_container, pred_config, wandb_run)

    # Write out files
    writer.write_experiment_output(
        csvdir, pred_analysis_data_container, reset_index=True, file_format=file_format
    )
    # Initialize reporter
    reporter = Reporter(logger=logger, wandb_run=wandb_run)

    # generate the report
    logger.info("Starting report generation.")
    reporter.create_report(pred_analysis_config, csvdir, figdir)


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
    parser = setup_rsmcmd_parser(
        "rsmtool",
        uses_output_directory=True,
        allows_overwriting=True,
        uses_subgroups=True,
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
        logger.info(f"Output directory: {args.output_dir}")
        run_experiment(
            abspath(args.config_file),
            abspath(args.output_dir),
            overwrite_output=args.force_write,
        )

    else:
        # when generating, log to stderr
        logging.root.addHandler(stderr_handler)

        # auto-generate an example configuration and print it to STDOUT
        generator = ConfigurationGenerator(
            "rsmtool",
            as_string=True,
            suppress_warnings=args.quiet,
            use_subgroups=args.subgroups,
        )
        configuration = (
            generator.interact(output_file_name=args.output_file.name if args.output_file else None)
            if args.interactive
            else generator.generate()
        )
        print(configuration, file=args.output_file)


if __name__ == "__main__":
    main()
