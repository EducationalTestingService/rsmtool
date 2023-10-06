#!/usr/bin/env python
"""
Explain a SKLL model using SHAP explainers.

:author: Remo Nitschke (rnitschke@ets.org)
:author: Zhaoyang Xie (zxie@etscanada.ca)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

import glob
import json
import logging
import os
import pickle
import sys
from os import listdir
from os.path import abspath, basename, exists, join, normpath, splitext
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from skll.data import FeatureSet

from .configuration_parser import configure
from .modeler import Modeler
from .preprocessor import FeaturePreprocessor
from .reader import DataReader
from .reporter import Reporter
from .utils.commandline import ConfigurationGenerator, setup_rsmcmd_parser
from .utils.constants import VALID_PARSER_SUBCOMMANDS
from .utils.conversion import parse_range
from .utils.logging import LogFormatter
from .utils.wandb import init_wandb_run, log_configuration_to_wandb


def select_examples(featureset, range_size=None):
    """
    Sample examples from the given featureset and return indices.

    Parameters
    ----------
    featureset: skll.data.FeatureSet
        The SKLL FeatureSet object from which we are sampling.
    range_size: Optional[Union[int, Tuple[int, int]]]
        A user defined sample size or range. If ``None``, all examples in the
        featureset are selected. If it's a size (int), that many examples are
        randomly selected. If it's a tuple, the two integers in the tuple
        define the size of the range of examples that is selected.

    Returns
    -------
    Dict[int, str]
        Dictionary mapping the position of the selected examples to their IDs.
    """
    fs_ids = featureset.ids
    if range_size is None:
        selected_ids = fs_ids
    elif isinstance(range_size, int):
        selected_ids = shap.sample(fs_ids, range_size)
    elif isinstance(range_size, tuple):
        selected_ids = np.array(range_size)
    else:
        start, end = range_size
        # NOTE: include the end index in the selected examples since it's more intuitive
        selected_ids = fs_ids[start : end + 1]  # noqa: E203

    # make sure that ``selected_ids`` is the same data type as ``fs_ids``
    selected_ids = selected_ids.astype(fs_ids.dtype)

    # find the positions of the selected ids in the original featureset
    try:
        selected_positions = [np.where(fs_ids == id_)[0][0] for id_ in selected_ids]
    except IndexError:
        raise ValueError(
            "Samples could not be selected; please check your configuration file."
        ) from None

    # create and return a dictionary mapping the position to the IDs
    return dict(zip(selected_positions, selected_ids))


def mask(learner, featureset, feature_range=None):
    """
    Sample examples from featureset used by learner.

    An example refers to a specific data instance in the data set.
    Selects examples based on either sub-sampling specific indices or randomly
    of a fixed size. Return the feature values for the selected examples as
    a numpy array.

    Parameters
    ----------
    learner : skll.learner.Learner
        SKLL Learner object that we wish to explain the predictions of.
    featureset : skll.data.FeatureSet
        SKLL FeatureSet object from which to sample examples.
    feature_range : Optional[int, Tuple[int, int]]
        If this is an integer, create a random sub-sample of that size. If this
        is a tuple, sub-sample the range of examples using the two values
        in the tuple. If this is ``None``, use all of the examples without
        any sub-sampling.

    Returns
    -------
    Dict[int, str]
        Dictionary mapping the position of the selected examples to their IDs.
        This is useful for figuring out which specific examples were selected.
    numpy.ndarray
        A 2D numpy array containing sampled feature rows.
    """
    # get a sparse matrix with the features that were actually used
    features = learner.feat_selector.transform(
        learner.feat_vectorizer.transform(
            featureset.vectorizer.inverse_transform(featureset.features)
        )
    )

    # sample examples from the featureset
    selected_feature_map = select_examples(featureset, range_size=feature_range)

    # if the user specified a sample size or a range, use it; otherwise all
    # features will be selected
    if feature_range:
        positions = list(selected_feature_map.keys())
        features = features[positions, :]

    # convert to a dense array if not already one
    features = features.toarray() if not isinstance(features, np.ndarray) else features
    return selected_feature_map, features


def generate_explanation(
    config_file_or_obj_or_dict,
    output_dir,
    overwrite_output=False,
    logger=None,
    wandb_run=None,
):
    """
    Generate a shap.Explanation object.

    This function does all the heavy lifting. It loads the model, creates an explainer, and
    generates an explanation object. It then calls generate_report() in order to generate a SHAP
    report.

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
    logger : Optional[logging object]
        A logging object. If ``None`` is passed, get logger from ``__name__``.
        Defaults to ``None``.
    wandb_run : wandb.Run
        A wandb run object that will be used to log artifacts and tables.
        If ``None`` is passed, a new wandb run will be initialized if
        wandb is enabled in the configuration. Defaults to ``None``.

    Raises
    ------
    FileNotFoundError
        If any file contained in ``config_file_or_obj_or_dict`` cannot be located.
    ValueError
        If both ``sample_range`` and ``sample_size`` are defined in the configuration
        file.


    """
    logger = logger if logger else logging.getLogger(__name__)

    # make sure all necessary directories exist
    os.makedirs(output_dir, exist_ok=True)
    csvdir = abspath(join(output_dir, "output"))
    figdir = abspath(join(output_dir, "figure"))
    reportdir = abspath(join(output_dir, "report"))

    os.makedirs(csvdir, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(reportdir, exist_ok=True)

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

    configuration = configure("rsmexplain", config_file_or_obj_or_dict)

    logger.info("Saving configuration file.")
    configuration.save(output_dir)

    # If wandb logging is enabled, and wandb_run is not provided,
    # start a wandb run and log configuration
    if wandb_run is None:
        wandb_run = init_wandb_run(configuration)
    log_configuration_to_wandb(wandb_run, configuration)

    # get the experiment ID
    experiment_id = configuration["experiment_id"]

    # check that only one of `sample_range`, `sample_size` or `sample_range` is specified
    has_sample_range = configuration.get("sample_range") is not None
    has_sample_size = configuration.get("sample_size") is not None
    has_sample_ids = configuration.get("sample_ids") is not None
    if sum([has_sample_range, has_sample_size, has_sample_ids]) > 1:
        raise ValueError(
            "You must specify one of 'sample_range', 'sample_size' or 'sample_ids'. "
            "Please refer to the `rsmexplain` documentation for more details. "
        )

    # find the rsmtool experiment directory
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
            f"The directory {experiment_output_dir} does not contain any rsmtool models."
        )

    experiment_ids = [splitext(basename(mf))[0] for mf in model_files]
    if experiment_id not in experiment_ids:
        raise FileNotFoundError(
            f"{experiment_output_dir} does not contain a model "
            f'for the experiment "{experiment_id}". The following '
            f"experiments are contained in this directory: {experiment_ids}"
        )

    # check that the directory contains the file with feature names and info
    expected_feature_file_name = f"{experiment_id}_feature.csv"
    if not exists(join(experiment_output_dir, expected_feature_file_name)):
        raise FileNotFoundError(
            f"{experiment_output_dir} does not contain the "
            f"required file {expected_feature_file_name} that was "
            f"generated during model training."
        )

    # read the original rsmtool configuration file, if it exists, and ensure
    # that we use its value of `standardize_features` and `truncate_outliers`
    # even if that means we have to override the values specified in the
    # rsmexplain configuration file
    expected_config_file_path = join(experiment_output_dir, f"{experiment_id}_rsmtool.json")
    if exists(expected_config_file_path):
        with open(expected_config_file_path, "r") as rsmtool_configfh:
            rsmtool_configuration = json.load(rsmtool_configfh)

        for option in ["standardize_features", "truncate_outliers"]:
            rsmtool_value = rsmtool_configuration[option]
            rsmexplain_value = configuration[option]
            if rsmexplain_value != rsmtool_value:
                logger.warning(
                    f"overwriting current `{option}` value "
                    f"({rsmexplain_value}) to match "
                    f"value specified in original rsmtool experiment "
                    f"({rsmtool_value})."
                )
                configuration[option] = rsmtool_value

    # if the original experiment rsmtool does not exist, let the user know
    else:
        logger.warning(
            "cannot locate original rsmtool configuration; "
            "ensure that the values of `standardize_features` "
            "and `truncate_outliers` were the same as when running rsmtool."
        )

    # load the background and explain data sets
    (background_data_path, explain_data_path) = DataReader.locate_files(
        [configuration["background_data"], configuration["explain_data"]],
        configuration.configdir,
    )

    if not background_data_path:
        raise FileNotFoundError(f"Input file {configuration['background_data']} does not exist")
    if not explain_data_path:
        raise FileNotFoundError(f"Input file {configuration['explain_data']} does not exist")

    # read the background data, explain data, and feature info files
    feature_info_path = join(experiment_output_dir, f"{experiment_id}_feature.csv")
    file_paths = [background_data_path, explain_data_path, feature_info_path]
    file_names = [
        "background_features",
        "explain_features",
        "feature_info",
    ]
    reader = DataReader(file_paths, file_names)
    container = reader.read(kwargs_dict={"feature_info": {"index_col": 0}})

    # ensure that the background data is large enough for meaningful explanations
    background_data_size = len(container.background_features)
    if background_data_size < 300:
        logger.error(
            f"The background data {background_data_path} contains only "
            f"{background_data_size} examples. It must contain at least 300 examples "
            "to ensure meaningful explanations."
        )
        sys.exit(1)

    # now pre-process the background and explain data features to match
    # what the model expects
    processor = FeaturePreprocessor(logger=logger)

    (_, processed_container) = processor.process_data(
        configuration, container, context="rsmexplain"
    )

    # create featuresets from pre-processed background and explain features
    background_fs = FeatureSet.from_data_frame(
        processed_container.background_features_preprocessed, "background"
    )
    explain_fs = FeatureSet.from_data_frame(
        processed_container.explain_features_preprocessed, "explain"
    )

    # get the SKLL learner object for the rsmtool experiment and its feature names
    modeler = Modeler.load_from_file(join(experiment_output_dir, f"{experiment_id}.model"))
    learner = modeler.learner
    feature_names = list(learner.get_feature_names_out())

    # compute the background kmeans distribution
    _, all_background_features = mask(learner, background_fs)
    background_distribution = shap.kmeans(
        all_background_features, configuration["background_kmeans_size"]
    )

    # get and parse the value of either the sample range or the sample size
    if has_sample_size:
        range_size = int(configuration.get("sample_size"))
    elif has_sample_range:
        range_size = parse_range(configuration.get("sample_range"))
    elif has_sample_ids:
        range_size = configuration.get("sample_ids").split(",")
        range_size = tuple([id_.strip() for id_ in range_size])
    else:
        range_size = None
        logger.warning(
            "Since 'sample_range', 'sample_size' and 'sample_ids' are all unspecified, "
            "explanations will be generated for the *entire* data set which "
            "could be very slow, depending on its size. "
        )

    # get the features we want to explain
    ids, data_features = mask(learner, explain_fs, feature_range=range_size)

    # define a shap explainer
    explainer = shap.explainers.Sampling(
        learner.model.predict,
        background_distribution,
        feature_names=feature_names,
        seed=np.random.seed(42),
    )

    logger.info(
        f"Generating SHAP explanations for {len(ids)} "
        f"examples from {configuration['explain_data']}"
    )
    explanation = explainer(data_features)

    # add feature names if they aren't already specified
    if explanation.feature_names is None:
        explanation.feature_names = feature_names

    # the explainer does not correctly generate base value arrays sometimes;
    # sometimes it's a single float or sometimes an array with a (1,) shape
    # so let's fix it if that happens
    base_values = explanation.base_values
    if not isinstance(base_values, np.ndarray):
        explanation.base_values = np.repeat(base_values, explanation.values.shape[0])

    # re-generate the explanation here, because manually munging the feature
    # names and base values can break some plots
    # TODO: check if this is still necessary in future versions of shap
    explanation = shap.Explanation(
        explanation.values,
        base_values=explanation.base_values,
        data=explanation.data,
        feature_names=explanation.feature_names,
    )

    # generate the HTML report
    generate_report(explanation, output_dir, ids, configuration, logger, wandb_run=wandb_run)


def generate_report(explanation, output_dir, ids, configuration, logger=None, wandb_run=None):
    """
    Generate an rsmexplain report.

    This function also saves a series of files to disk, including
    pickled versions of the explanation object and the ID dictionary.
    All SHAP values are also saved as CSV files.

    Parameters
    ----------
    explanation: shap.Explanation
        SHAP explanation object containing SHAP values, data points, feature
        names and base values.
    output_dir : str
        Path to the experiment output directory.
    ids: dict
        Dictionary mapping new row indices to original FeatureSet ids.
    configuration: rsmtool.configuration_parser.Configuration
        The Configuration object for rsmexplain.
    logger : logging object, optional
        A logging object. If ``None`` is passed, get logger from ``__name__``.
        Defaults to ``None``.
    wandb_run : wandb.Run
        A wandb run object that will be used to log artifacts and tables.
        If ``None`` is passed, a new wandb run will be initialized if
        wandb is enabled in the configuration. Defaults to ``None``.
    """
    logger = logger if logger else logging.getLogger(__name__)

    # get the various output sub-directories which should already exist
    csvdir = abspath(join(output_dir, "output"))
    reportdir = abspath(join(output_dir, "report"))

    # get the experiment ID
    experiment_id = configuration["experiment_id"]

    # first write the explanation object to disk, in case we need it later
    explanation_path = join(csvdir, f"{experiment_id}_explanation.pkl")
    with open(explanation_path, "wb") as pickle_out:
        pickle.dump(explanation, pickle_out)
    configuration["explanation"] = explanation_path

    id_path = join(csvdir, f"{experiment_id}_ids.pkl")
    with open(id_path, "wb") as pickle_out:
        pickle.dump(ids, pickle_out)
    configuration["ids"] = id_path

    # create various versions of the SHAP values to write to disk
    csv_path = join(csvdir, f"{experiment_id}_shap_values.csv")
    shap_frame = pd.DataFrame(
        explanation.values, columns=explanation.feature_names, index=ids.values()
    )
    shap_frame.to_csv(csv_path)

    # compute the various absolute value variants of the SHAP values and
    # write out that dataframe to disk.
    csv_path_abs = join(csvdir, f"{experiment_id}_absolute_shap_values.csv")
    df_abs = pd.DataFrame(
        [shap_frame.abs().mean(), shap_frame.abs().max(), shap_frame.abs().min()],
        index=["abs. mean shap", "abs. max shap", "abs. min shap"],
    ).transpose()
    df_abs.to_csv(csv_path_abs, index_label="")

    # Initialize a reporter instance and add the sections:
    reporter = Reporter(logger=logger, wandb_run=wandb_run)
    general_report_sections = configuration["general_sections"]
    special_report_sections = configuration["special_sections"]

    # get any custom sections and locate them to make sure
    # that they exist, otherwise raise an exception
    custom_report_section_paths = configuration["custom_sections"]

    if custom_report_section_paths:
        logger.info("Locating custom report sections")
        custom_report_sections = Reporter.locate_custom_sections(
            custom_report_section_paths, configuration.configdir
        )
    else:
        custom_report_sections = []

    # leverage custom sections to allow users to turn `show_auto_cohorts` on and off
    notebooks_path = Path(__file__).parent / "notebooks"
    notebooks_path = notebooks_path.resolve()
    explanation_notebooks_path = notebooks_path / "explanations"

    # check to see whether a single or multiple examples have been chosen
    has_single_example = len(explanation.values) <= 1
    configuration["has_single_example"] = has_single_example

    # auto cohort plots will be displayed with more than one example selected
    if configuration["show_auto_cohorts"] and not has_single_example:
        custom_report_sections.append(f"{explanation_notebooks_path}/auto_cohorts.ipynb")

    # define all of the chosen notebook sections
    chosen_notebook_files = reporter.get_ordered_notebook_files(
        general_report_sections,
        special_report_sections,
        custom_report_sections,
        context="rsmexplain",
    )

    # add chosen notebook files to configuration and generate the report
    configuration["chosen_notebook_files"] = chosen_notebook_files
    reporter.create_explanation_report(configuration, csvdir, reportdir)


def main():
    """Run rsmexplain and generate explanation reports."""
    # set up the basic logging configuration
    formatter = LogFormatter()

    # need two handlers, one that prints to stdout for the "run" command and one that prints to
    # stderr from the "generate" command; the latter is important because do not want the warning
    # to show up in the generated configuration file
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)

    logging.root.setLevel(logging.INFO)
    logger = logging.getLogger(__name__)

    # set up an argument parser via our helper function
    parser = setup_rsmcmd_parser("rsmexplain", uses_output_directory=True, allows_overwriting=True)

    # if no arguments provided then just show the help message
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

    if args.subcommand == "run":
        # when running, log to stdout
        logging.root.addHandler(stdout_handler)

        # run the experiment
        logger.info(f"Output directory: {args.output_dir}")

        generate_explanation(
            abspath(args.config_file),
            abspath(args.output_dir),
            overwrite_output=args.force_write,
        )

    else:
        # when generating, log to stderr
        logging.root.addHandler(stderr_handler)

        # auto-generate an example configuration and print it to STDOUT
        generator = ConfigurationGenerator(
            "rsmexplain",
            as_string=True,
            suppress_warnings=args.quiet,
            use_subgroups=False,
        )
        configuration = (
            generator.interact(output_file_name=args.output_file.name if args.output_file else None)
            if args.interactive
            else generator.generate()
        )
        print(configuration, file=args.output_file)


if __name__ == "__main__":
    main()
