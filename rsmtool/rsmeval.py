#!/usr/bin/env python
"""
Run evaluation only experiments.

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

import logging
import os
import sys
from os import listdir
from os.path import abspath, exists, join

from .analyzer import Analyzer
from .configuration_parser import configure
from .preprocessor import FeaturePreprocessor
from .reader import DataReader
from .reporter import Reporter
from .utils.commandline import ConfigurationGenerator, setup_rsmcmd_parser
from .utils.constants import VALID_PARSER_SUBCOMMANDS
from .utils.logging import LogFormatter
from .utils.wandb import init_wandb_run, log_configuration_to_wandb
from .writer import DataWriter


def run_evaluation(
    config_file_or_obj_or_dict, output_dir, overwrite_output=False, logger=None, wandb_run=None
):
    """
    Run an rsmeval experiment using the given configuration.

    All outputs are generated under ``output_dir``. If ``overwrite_output``
    is ``True``, any existing output in ``output_dir`` is overwritten.

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
    """
    logger = logger if logger else logging.getLogger(__name__)

    # create the 'output' and the 'figure' sub-directories
    # where all the experiment output such as the CSV files
    # and the box plots will be saved
    csvdir = abspath(join(output_dir, "output"))
    figdir = abspath(join(output_dir, "figure"))
    reportdir = abspath(join(output_dir, "report"))
    os.makedirs(csvdir, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(reportdir, exist_ok=True)

    # Raise an error if the specified output directory
    # already contains a non-empty `output` directory, unless
    # ``overwrite_output`` was specified, in which case we assume
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
                f"The generated report might contain unexpected information "
                f"from a previous experiment."
            )

    configuration = configure("rsmeval", config_file_or_obj_or_dict)

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

    # Make sure prediction file can be located
    if not DataReader.locate_files(configuration["predictions_file"], configuration.configdir):
        raise FileNotFoundError(
            f"Error: Predictions file {configuration['predictions_file']} " f"not found."
        )

    scale_with = configuration.get("scale_with")

    # scale_with can be one of the following:
    # (a) 'raw' or None : the predictions are assumed to be 'raw' and should be used as is
    #                     when computing the metrics; the names for the final columns are
    #                     'raw', 'raw_trim' and 'raw_trim_round'.
    # (b) 'asis'        : the predictions are assumed to be pre-scaled and should be used as is
    #                     when computing the metrics; the names for the final columns are
    #                     'scale', 'scale_trim' and 'scale_trim_round'.
    # (c) a CSV file    : the predictions are assumed to be 'raw' and should be scaled
    #                     before computing the metrics; the names for the final columns are
    #                     'scale', 'scale_trim' and 'scale_trim_round'.

    # Check whether we want to do scaling
    do_scaling = scale_with is not None and scale_with not in ["asis", "raw"]

    # The paths to files and names for data container properties
    paths = ["predictions_file"]
    names = ["predictions"]

    # If we want to do scaling, get the scale file
    if do_scaling:
        # Make sure scale file can be located
        scale_file_location = DataReader.locate_files(scale_with, configuration.configdir)
        if not scale_file_location:
            raise FileNotFoundError(f"Could not find scaling file {scale_file_location}.")

        paths.append("scale_with")
        names.append("scale")

    # Get the paths, names, and converters for the DataReader
    (file_names, file_paths) = configuration.get_names_and_paths(paths, names)

    file_paths = DataReader.locate_files(file_paths, configuration.configdir)

    converters = {"predictions": configuration.get_default_converter()}

    logger.info(f"Reading predictions: {configuration['predictions_file']}.")

    # Initialize the reader
    reader = DataReader(file_paths, file_names, converters)
    data_container = reader.read()

    logger.info("Preprocessing predictions.")

    # Initialize the processor
    processor = FeaturePreprocessor(logger=logger)

    (processed_config, processed_container) = processor.process_data(
        configuration, data_container, context="rsmeval"
    )

    logger.info("Saving pre-processed predictions and metadata to disk.")
    writer.write_experiment_output(
        csvdir,
        processed_container,
        new_names_dict={
            "pred_test": "pred_processed",
            "test_excluded": "test_excluded_responses",
        },
        file_format=file_format,
    )

    # Initialize the analyzer
    analyzer = Analyzer(logger=logger)

    # do the data composition stats
    (
        analyzed_config,
        analyzed_container,
    ) = analyzer.run_data_composition_analyses_for_rsmeval(processed_container, processed_config)
    # Write out files
    writer.write_experiment_output(csvdir, analyzed_container, file_format=file_format)

    for_pred_data_container = analyzed_container + processed_container

    # run the analyses on the predictions of the model`
    logger.info("Running analyses on predictions.")
    (
        pred_analysis_config,
        pred_analysis_data_container,
    ) = analyzer.run_prediction_analyses(
        for_pred_data_container, analyzed_config, wandb_run=wandb_run
    )

    writer.write_experiment_output(
        csvdir, pred_analysis_data_container, reset_index=True, file_format=file_format
    )

    # Initialize reporter
    reporter = Reporter(logger=logger, wandb_run=wandb_run)

    # generate the report
    logger.info("Starting report generation.")
    reporter.create_report(pred_analysis_config, csvdir, figdir, context="rsmeval")


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
        "rsmeval",
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
        run_evaluation(
            abspath(args.config_file),
            abspath(args.output_dir),
            overwrite_output=args.force_write,
        )

    else:
        # when generating, log to stderr
        logging.root.addHandler(stderr_handler)

        # auto-generate an example configuration and print it to STDOUT
        generator = ConfigurationGenerator(
            "rsmeval",
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
