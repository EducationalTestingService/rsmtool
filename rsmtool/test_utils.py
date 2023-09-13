"""Utility functions for RSMTool tests."""
import os
import re
import sys
import warnings
from ast import literal_eval as eval
from filecmp import clear_cache, dircmp
from glob import glob
from importlib.machinery import SourceFileLoader
from inspect import getmembers, getsource, getsourcelines, isclass, isfunction
from os import remove
from os.path import basename, exists, join
from pathlib import Path
from shutil import copyfile, copytree, rmtree

import numpy as np
from bs4 import BeautifulSoup
from pandas.testing import assert_frame_equal

from .modeler import Modeler
from .reader import DataReader
from .rsmcompare import run_comparison
from .rsmeval import run_evaluation
from .rsmexplain import generate_explanation
from .rsmpredict import compute_and_save_predictions
from .rsmsummarize import run_summary
from .rsmtool import run_experiment
from .rsmxval import run_cross_validation

html_error_regexp = re.compile(r"Traceback \(most recent call last\)")
html_warning_regexp = re.compile(r'<div class=".*?output_stderr.*?>([^<]+)')
section_regexp = re.compile(r"<h2>(.*?)</h2>")

# get the directory containing the tests
rsmtool_test_dir = Path(__file__).absolute().parent.parent.joinpath("tests")

tools_with_input_data = ["rsmsummarize", "rsmcompare"]
tools_with_output = ["rsmtool", "rsmeval", "rsmsummarize", "rsmpredict", "rsmxval", "rsmexplain"]

# check if tests are being run in strict mode
# if so, any warnings found in HTML
# reports should not be ignored
STRICT_MODE = os.environ.get("STRICT", None)
IGNORE_WARNINGS = False if STRICT_MODE else True


def check_run_experiment(
    source,
    experiment_id,
    subgroups=None,
    consistency=False,
    skll=False,
    file_format="csv",
    given_test_dir=None,
    config_obj_or_dict=None,
    suppress_warnings_for=[],
):
    """
    Run a parameterized rsmtool experiment test.

    Parameters
    ----------
    source : str
        The name of the source directory containing the experiment
        configuration.
    experiment_id : str
        The experiment ID of the experiment.
    subgroups : list of str, optional
        List of subgroup names used in the experiment. If specified,
        outputs pertaining to subgroups are also checked as part of the
        test.
        Defaults to ``None``.
    consistency : bool, optional
        Whether to check consistency files as part of the experiment test.
        Generally, this should be true if the second human score column is
        specified.
        Defaults to ``False``.
    skll : bool, optional
        Whether the model being used in the experiment is a SKLL model
        in which case the coefficients, predictions, etc. will not be
        checked since they can vary across machines, due to parameter tuning.
        Defaults to ``False``.
    file_format : str, optional
        Which file format is being used for the output files of the experiment.
        Defaults to "csv".
    given_test_dir : str, optional
        Path where the test experiments are located. Unless specified, the
        rsmtool test directory is used. This can be useful when using these
        experiments to run tests for RSMExtra.
        Defaults to ``None``.
    config_obj_or_dict: configuration_parser.Configuration or dict, optional
        Configuration object or dictionary to use as an input, if any.
        If ``None``, the function will construct a path to the config file
        using ``source`` and ``experiment_id``.
    suppress_warnings_for : list, optional
        Categories for which warnings should be suppressed when running the
        experiments.
        Defaults to ``[]``.
    """
    # use the test directory from this file unless it's been overridden
    test_dir = given_test_dir if given_test_dir else rsmtool_test_dir

    if config_obj_or_dict is None:
        config_input = join(test_dir, "data", "experiments", source, f"{experiment_id}.json")
    else:
        config_input = config_obj_or_dict

    model_type = "skll" if skll else "rsmtool"

    do_run_experiment(
        source, experiment_id, config_input, suppress_warnings_for=suppress_warnings_for
    )

    output_dir = join("test_outputs", source, "output")
    expected_output_dir = join(test_dir, "data", "experiments", source, "output")
    html_report = join("test_outputs", source, "report", f"{experiment_id}_report.html")

    output_files = glob(join(output_dir, f"*.{file_format}"))
    for output_file in output_files:
        output_filename = basename(output_file)
        expected_output_file = join(expected_output_dir, output_filename)

        if exists(expected_output_file):
            check_file_output(output_file, expected_output_file, file_format=file_format)

    check_generated_output(output_files, experiment_id, model_type, file_format=file_format)

    if not skll:
        check_scaled_coefficients(output_dir, experiment_id, file_format=file_format)

    if subgroups:
        check_subgroup_outputs(output_dir, experiment_id, subgroups, file_format=file_format)

    if consistency:
        check_consistency_files_exist(output_files, experiment_id, file_format=file_format)

    # check report for any errors but ignore warnings
    # which we check below separately
    check_report(html_report, raise_warnings=False)

    # make sure that there are no warnings in the report
    # but ignore warnings if appropriate
    if not IGNORE_WARNINGS:
        warning_msgs = collect_warning_messages_from_report(html_report)
        assert len(warning_msgs) == 0


def check_run_evaluation(
    source,
    experiment_id,
    subgroups=None,
    consistency=False,
    file_format="csv",
    given_test_dir=None,
    config_obj_or_dict=None,
    suppress_warnings_for=[],
):
    """
    Run a parameterized rsmeval experiment test.

    Parameters
    ----------
    source : str
        The name of the source directory containing the experiment
        configuration.
    experiment_id : str
        The experiment ID of the experiment.
    subgroups : list of str, optional
        List of subgroup names used in the experiment. If specified,
        outputs pertaining to subgroups are also checked as part of the
        test.
        Defaults to ``None``.
    consistency : bool, optional
        Whether to check consistency files as part of the experiment test.
        Generally, this should be true if the second human score column is
        specified.
        Defaults to ``False``.
    file_format : str, optional
        Which file format is being used for the output files of the experiment.
        Defaults to "csv".
    given_test_dir : str, optional
        Path where the test experiments are located. Unless specified, the
        rsmtool test directory is used. This can be useful when using these
        experiments to run tests for RSMExtra.
        Defaults to ``None``.
    config_obj_or_dict: configuration_parser.Configuration or dict, optional
        Configuration object or dictionary to use as an input, if any.
        If ``None``, the function will construct a path to the config file
        using ``source`` and ``experiment_id``.
        Defaults to ``None``.
    suppress_warnings_for : list, optional
        Categories for which warnings should be suppressed when running the
        experiments.
        Defaults to ``[]``.
    """
    # use the test directory from this file unless it's been overridden
    test_dir = given_test_dir if given_test_dir else rsmtool_test_dir

    if config_obj_or_dict is None:
        config_input = join(test_dir, "data", "experiments", source, f"{experiment_id}.json")
    else:
        config_input = config_obj_or_dict

    do_run_evaluation(
        source, experiment_id, config_input, suppress_warnings_for=suppress_warnings_for
    )

    output_dir = join("test_outputs", source, "output")
    expected_output_dir = join(test_dir, "data", "experiments", source, "output")
    html_report = join("test_outputs", source, "report", f"{experiment_id}_report.html")

    output_files = glob(join(output_dir, f"*.{file_format}"))
    for output_file in output_files:
        output_filename = basename(output_file)
        expected_output_file = join(expected_output_dir, output_filename)

        if exists(expected_output_file):
            check_file_output(output_file, expected_output_file, file_format=file_format)

    if consistency:
        check_consistency_files_exist(output_files, experiment_id)

    # check report for any errors but ignore warnings
    # which we check below separately
    check_report(html_report, raise_warnings=False)

    # make sure that there are no warnings in the report
    # but ignore warnings if appropriate
    if not IGNORE_WARNINGS:
        warning_msgs = collect_warning_messages_from_report(html_report)
        assert len(warning_msgs) == 0


def check_run_explain(
    source,
    experiment_id,
    file_format="csv",
    given_test_dir=None,
    config_obj_or_dict=None,
    suppress_warnings_for=[],
):
    """
    Run a parameterized rsmexplain experiment test.

    Parameters
    ----------
    source : str
        The name of the source directory containing the experiment
        configuration.
    experiment_id : str
        The experiment ID of the experiment.
    given_test_dir : str, optional
        Path where the test experiments are located. Unless specified, the
        rsmtool test directory is used. This can be useful when using these
        experiments to run tests for RSMExtra.
        Defaults to ``None``.
    config_obj_or_dict: configuration_parser.Configuration or dict, optional
        Configuration object or dictionary to use as an input.
        If ``None``, the function will construct a path to the config file
        using ``source`` and ``experiment_id``.
        Defaults to ``None``.
    suppress_warnings_for : list, optional
        Categories for which warnings should be suppressed when running the
        experiments.
        Defaults to ```[]``.
    """
    # use the test directory from this file unless it's been overridden
    test_dir = given_test_dir if given_test_dir else rsmtool_test_dir

    if config_obj_or_dict is None:
        config_input = join(test_dir, "data", "experiments", source, "rsmexplain.json")
    else:
        config_input = config_obj_or_dict

    do_run_explain(source, config_input, suppress_warnings_for=suppress_warnings_for)

    html_report = join("test_outputs", source, "report", f"{experiment_id}_explain_report.html")

    # check report for any errors but ignore warnings
    # which we check below separately
    check_report(html_report, raise_warnings=False)

    output_dir = join("test_outputs", source, "output")
    expected_output_dir = join(test_dir, "data", "experiments", source, "output")

    output_files = glob(join(output_dir, f"*.{file_format}"))
    for output_file in output_files:
        output_filename = basename(output_file)
        expected_output_file = join(expected_output_dir, output_filename)

        if exists(expected_output_file):
            check_file_output(output_file, expected_output_file, file_format=file_format)

    # make sure that there are no warnings in the report
    # but ignore warnings if appropriate
    if not IGNORE_WARNINGS:
        warning_msgs = collect_warning_messages_from_report(html_report)
        assert len(warning_msgs) == 0


def check_run_comparison(
    source,
    experiment_id,
    given_test_dir=None,
    config_obj_or_dict=None,
    suppress_warnings_for=[],
):
    """
    Run a parameterized rsmcompare experiment test.

    Parameters
    ----------
    source : str
        The name of the source directory containing the experiment
        configuration.
    experiment_id : str
        The experiment ID of the experiment.
    given_test_dir : str, optional
        Path where the test experiments are located. Unless specified, the
        rsmtool test directory is used. This can be useful when using these
        experiments to run tests for RSMExtra.
        Defaults to ``None``.
    config_obj_or_dict: configuration_parser.Configuration or dict, optional
        Configuration object or dictionary to use as an input.
        If ``None``, the function will construct a path to the config file
        using ``source`` and ``experiment_id``.
        Defaults to ``None``.
    suppress_warnings_for : list, optional
        Categories for which warnings should be suppressed when running the
        experiments.
        Defaults to ```[]``.
    """
    # use the test directory from this file unless it's been overridden
    test_dir = given_test_dir if given_test_dir else rsmtool_test_dir

    if config_obj_or_dict is None:
        config_input = join(test_dir, "data", "experiments", source, "rsmcompare.json")
    else:
        config_input = config_obj_or_dict

    do_run_comparison(source, config_input, suppress_warnings_for=suppress_warnings_for)

    html_report = join("test_outputs", source, f"{experiment_id}_report.html")

    # check report for any errors but ignore warnings
    # which we check below separately
    check_report(html_report, raise_warnings=False)

    # make sure that there are no warnings in the report
    # but ignore warnings if appropriate
    if not IGNORE_WARNINGS:
        warning_msgs = collect_warning_messages_from_report(html_report)
        assert len(warning_msgs) == 0


def check_run_prediction(
    source,
    excluded=False,
    file_format="csv",
    given_test_dir=None,
    config_obj_or_dict=None,
    suppress_warnings_for=[],
):
    """
    Run a parameterized rsmpredict experiment test.

    Parameters
    ----------
    source : str
        The name of the source directory containing the experiment
        configuration.
    excluded : bool, optional
        Whether to check the excluded responses file as part of the test.
        Defaults to ``False``.
    file_format : str, optional
        Which file format is being used for the output files of the experiment.
        Defaults to "csv".
    given_test_dir : str, optional
        Path where the test experiments are located. Unless specified, the
        rsmtool test directory is used. This can be useful when using these
        experiments to run tests for RSMExtra.
        Defaults to ``None``.
    config_obj_or_dict: configuration_parser.Configuration or dict, optional
        Configuration object or dictionary to use as an input.
        If ``None``, the function will construct a path to the config file
        using ``source`` and ``experiment_id``.
        Defaults to ``None``.
    suppress_warnings_for : list, optional
        Categories for which warnings should be suppressed when running the
        experiments.
        Defaults to ``[]``.
    """
    # use the test directory from this file unless it's been overridden
    test_dir = given_test_dir if given_test_dir else rsmtool_test_dir

    if config_obj_or_dict is None:
        config_input = join(test_dir, "data", "experiments", source, "rsmpredict.json")
    else:
        config_input = config_obj_or_dict

    do_run_prediction(source, config_input, suppress_warnings_for=suppress_warnings_for)

    output_dir = join("test_outputs", source, "output")
    expected_output_dir = join(test_dir, "data", "experiments", source, "output")

    output_files = [
        f"predictions.{file_format}",
        f"preprocessed_features.{file_format}",
    ]
    if excluded:
        output_files.append(f"predictions_excluded_responses.{file_format}")
    for output_file in output_files:
        generated_output_file = join(output_dir, output_file)
        expected_output_file = join(expected_output_dir, output_file)

        check_file_output(generated_output_file, expected_output_file)


def check_run_summary(
    source,
    file_format="csv",
    given_test_dir=None,
    config_obj_or_dict=None,
    suppress_warnings_for=[],
):
    """
    Run a parameterized rsmsummarize experiment test.

    Parameters
    ----------
    source : str
        The name of the source directory containing the experiment
        configuration.
    file_format : str, optional
        Which file format is being used for the output files of the experiment.
        Defaults to "csv".
    given_test_dir : str, optional
        Path where the test experiments are located. Unless specified, the
        rsmtool test directory is used. This can be useful when using these
        experiments to run tests for RSMExtra.
        Defaults to ``None``.
    config_obj_or_dict: configuration_parser.Configuration or dict, optional
        Configuration object or dictionary to use as an input.
        If ``None``, the function will construct a path to the config file
        using ``source`` and ``experiment_id``.
        Defaults to ``None``.
    suppress_warnings_for : list, optional
        Categories for which warnings should be suppressed when running the
        experiments.
        Defaults to ``[]``.
    """
    # use the test directory from this file unless it's been overridden
    test_dir = given_test_dir if given_test_dir else rsmtool_test_dir

    if config_obj_or_dict is None:
        config_input = join(test_dir, "data", "experiments", source, "rsmsummarize.json")
    else:
        config_input = config_obj_or_dict

    do_run_summary(source, config_input, suppress_warnings_for=suppress_warnings_for)

    html_report = join("test_outputs", source, "report", "model_comparison_report.html")

    output_dir = join("test_outputs", source, "output")
    expected_output_dir = join(test_dir, "data", "experiments", source, "output")

    output_files = glob(join(output_dir, f"*.{file_format}"))
    for output_file in output_files:
        output_filename = basename(output_file)
        expected_output_file = join(expected_output_dir, output_filename)

        if exists(expected_output_file):
            check_file_output(output_file, expected_output_file)

    # check report for any errors but ignore warnings
    # which we check below separately
    check_report(html_report, raise_warnings=False)

    # make sure that there are no warnings in the report
    # but ignore warnings if appropriate
    if not IGNORE_WARNINGS:
        warning_msgs = collect_warning_messages_from_report(html_report)
        assert len(warning_msgs) == 0


def check_run_cross_validation(
    source,
    experiment_id,
    folds=5,
    subgroups=None,
    consistency=False,
    skll=False,
    file_format="csv",
    given_test_dir=None,
    config_obj_or_dict=None,
    suppress_warnings_for=[],
):
    """
    Run a parameterized rsmxval experiment test.

    Parameters
    ----------
    source : str
        The name of the source directory containing the experiment
        configuration.
    experiment_id : str
        The experiment ID of the experiment.
    folds : int, optional
        Number of folds being used in the cross-validation experiment.
        Defaults to 5.
    subgroups : list of str, optional
        List of subgroup names used in the experiment. If specified,
        outputs pertaining to subgroups are also checked as part of the
        test.
        Defaults to ``None``.
    consistency : bool, optional
        Whether to check consistency files as part of the experiment test.
        Generally, this should be true if the second human score column is
        specified.
        Defaults to ``False``.
    skll : bool, optional
        Whether the model being used in the experiment is a SKLL model
        in which case the coefficients, predictions, etc. will not be
        checked since they can vary across machines, due to parameter tuning.
        Defaults to ``False``.
    file_format : str, optional
        Which file format is being used for the output files of the experiment.
        Defaults to "csv".
    given_test_dir : str, optional
        Path where the test experiments are located. Unless specified, the
        rsmtool test directory is used. This can be useful when using these
        experiments to run tests for RSMExtra.
        Defaults to ``None``.
    config_obj_or_dict : configuration_parser.Configuration or dict, optional
        Configuration object or dictionary to use as an input, if any.
        If ``None``, the function will construct a path to the config file
        using ``source`` and ``experiment_id``.
    suppress_warnings_for : list, optional
        Categories for which warnings should be suppressed when running the
        experiments.
        Defaults to ``[]``.
    """
    # use the test directory from this file unless it's been overridden
    test_dir = given_test_dir if given_test_dir else rsmtool_test_dir

    if config_obj_or_dict is None:
        config_input = join(test_dir, "data", "experiments", source, f"{experiment_id}.json")
    else:
        config_input = config_obj_or_dict

    model_type = "skll" if skll else "rsmtool"

    do_run_cross_validation(
        source, experiment_id, config_input, suppress_warnings_for=suppress_warnings_for
    )

    output_prefix = join("test_outputs", source)
    expected_output_prefix = join(test_dir, "data", "experiments", source, "output")

    # first check that each fold's rsmtool output is as expected
    actual_folds_dir = join(output_prefix, "folds")
    expected_folds_dir = join(expected_output_prefix, "folds")
    for fold_num in range(1, folds + 1):
        fold_experiment_id = f"{experiment_id}_fold{fold_num:02}"
        fold_output_dir = join(actual_folds_dir, f"{fold_num:02}", "output")
        fold_output_files = glob(join(fold_output_dir, f"*.{file_format}"))
        for fold_output_file in fold_output_files:
            output_filename = basename(fold_output_file)
            expected_output_file = join(
                expected_folds_dir, f"{fold_num:02}", "output", output_filename
            )

            if exists(expected_output_file):
                check_file_output(fold_output_file, expected_output_file, file_format=file_format)

        check_generated_output(
            fold_output_files, fold_experiment_id, model_type, file_format=file_format
        )

        if not skll:
            check_scaled_coefficients(fold_output_dir, fold_experiment_id, file_format=file_format)

        if subgroups:
            check_subgroup_outputs(
                fold_output_dir, fold_experiment_id, subgroups, file_format=file_format
            )

        if consistency:
            check_consistency_files_exist(
                fold_output_files, fold_experiment_id, file_format=file_format
            )

    # next check that the evaluation output is as expected
    actual_eval_output_dir = join(output_prefix, "evaluation", "output")
    expected_eval_output_dir = join(expected_output_prefix, "evaluation", "output")

    output_files = glob(join(actual_eval_output_dir, f"*.{file_format}"))
    for output_file in output_files:
        output_filename = basename(output_file)
        expected_output_file = join(expected_eval_output_dir, output_filename)

        if exists(expected_output_file):
            check_file_output(output_file, expected_output_file, file_format=file_format)

    if consistency:
        check_consistency_files_exist(output_files, f"{experiment_id}_evaluation")

    # next check that the summary output is as expected
    actual_summary_output_dir = join(output_prefix, "fold-summary", "output")
    expected_summary_output_dir = join(expected_output_prefix, "fold-summary", "output")

    output_files = glob(join(actual_summary_output_dir, f"*.{file_format}"))
    for output_file in output_files:
        output_filename = basename(output_file)
        expected_output_file = join(expected_summary_output_dir, output_filename)

        if exists(expected_output_file):
            check_file_output(output_file, expected_output_file)

    # next check that the final model rsmtool output is as expected
    actual_final_model_output_dir = join(output_prefix, "final-model", "output")
    expected_final_model_output_dir = join(expected_output_prefix, "final-model", "output")
    model_experiment_id = f"{experiment_id}_model"

    output_files = glob(join(actual_final_model_output_dir, f"*.{file_format}"))
    for output_file in output_files:
        output_filename = basename(output_file)
        expected_output_file = join(expected_final_model_output_dir, output_filename)

        if exists(expected_output_file):
            check_file_output(output_file, expected_output_file, file_format=file_format)

    check_generated_output(output_files, model_experiment_id, model_type, file_format=file_format)

    if not skll:
        check_scaled_coefficients(
            actual_final_model_output_dir, model_experiment_id, file_format=file_format
        )

    if subgroups:
        check_subgroup_outputs(
            actual_final_model_output_dir,
            model_experiment_id,
            subgroups,
            file_format=file_format,
        )

    # finally check all the HTML reports for any errors but ignore warnings
    # which we check below separately
    per_fold_html_reports = glob(join(output_prefix, "folds", "*", "report", "*.html"))

    evaluation_report = join(
        output_prefix, "evaluation", "report", f"{experiment_id}_evaluation_report.html"
    )

    summary_report = join(
        output_prefix,
        "fold-summary",
        "report",
        f"{experiment_id}_fold_summary_report.html",
    )

    final_model_report = join(
        output_prefix, "final-model", "report", f"{experiment_id}_model_report.html"
    )

    for html_report in per_fold_html_reports + [
        evaluation_report,
        summary_report,
        final_model_report,
    ]:
        check_report(html_report, raise_warnings=False)

        # make sure that there are no warnings in the report
        # but ignore warnings if appropriate
        if not IGNORE_WARNINGS:
            warning_msgs = collect_warning_messages_from_report(html_report)
            assert len(warning_msgs) == 0


def do_run_experiment(source, experiment_id, config_input, suppress_warnings_for=[]):
    """
    Run rsmtool experiment automatically.

    Use the given experiment configuration file located in the
    given source directory and use the given experiment ID.

    Parameters
    ----------
    source : str
        Path to where the test experiment is located on disk.
    experiment_id : str
        Experiment ID to use when running.
    config_input : str or Configuration or dict
        Path to the experiment configuration file,
        or a ``configuration_parser.Configuration`` object
        or a dictionary with keys corresponding to fields in the
        configuration file.
    suppress_warnings_for : list, optional
        Categories for which warnings should be suppressed
        when running the experiments. Note that ``RuntimeWarning``s
        are always suppressed.
        Defaults to ``[]``.
    """
    source_output_dir = "test_outputs"
    experiment_dir = join(source_output_dir, source)

    # remove all previously created files
    for output_subdir in ["output", "figure", "report"]:
        files = glob(join(source_output_dir, source, output_subdir, "*"))
        for f in files:
            remove(f)

    with warnings.catch_warnings():
        # always suppress runtime warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # suppress additional warning types if specified
        for warning_type in suppress_warnings_for:
            warnings.filterwarnings("ignore", category=warning_type)

        run_experiment(config_input, experiment_dir)


def do_run_evaluation(source, experiment_id, config_input, suppress_warnings_for=[]):
    """
    Run rsmeval experiment automatically.

    Use the given experiment configuration file located in the given
    source directory and use the given experiment ID.

    Parameters
    ----------
    source : str
        Path to where the test experiment is located on disk.
    experiment_id : str
        Experiment ID to use when running.
    config_input : str or Configuration or dict
        Path to the experiment configuration file,
        or a ``configuration_parser.Configuration`` object,
        or a dictionary with keys corresponding to fields in the
        configuration file.
    suppress_warnings_for : list, optional
        Categories for which warnings should be suppressed
        when running the experiments. Note that ``RuntimeWarning``s
        are always suppressed.
        Defaults to ``[]``.
    """
    source_output_dir = "test_outputs"
    experiment_dir = join(source_output_dir, source)

    # remove all previously created files
    for output_subdir in ["output", "figure", "report"]:
        files = glob(join(source_output_dir, source, output_subdir, "*"))
        for f in files:
            remove(f)

    with warnings.catch_warnings():
        # always suppress runtime warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # suppress additional warning types if specified
        for warning_type in suppress_warnings_for:
            warnings.filterwarnings("ignore", category=warning_type)

        run_evaluation(config_input, experiment_dir)


def do_run_explain(source, config_input, suppress_warnings_for=[]):
    """
    Run rsmexplain experiment automatically.

    Use the given experiment configuration file located in the given
    source directory.

    Parameters
    ----------
    source : str
        Path to where the test experiment is located on disk.
    config_input : str or Configuration or dict
        Path to the experiment configuration file,
        or a ``configuration_parser.Configuration`` object,
        or a dictionary with keys corresponding to fields in the
        configuration file
    suppress_warnings_for : list, optional
        Categories for which warnings should be suppressed when running the
        experiments.
        Defaults to ``[]``.
    """
    source_output_dir = "test_outputs"
    experiment_dir = join(source_output_dir, source)

    # remove all previously created files
    for output_subdir in ["output", "figure", "report"]:
        files = glob(join(source_output_dir, source, output_subdir, "*"))
        for f in files:
            remove(f)

    with warnings.catch_warnings():
        # always suppress runtime warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # suppress additional warning types if specified
        for warning_type in suppress_warnings_for:
            warnings.filterwarnings("ignore", category=warning_type)

        generate_explanation(config_input, experiment_dir)


def do_run_prediction(source, config_input, suppress_warnings_for=[]):
    """
    Run rsmpredict experiment automatically.

    Use the given experiment configuration file located in the given
    source directory.

    Parameters
    ----------
    source : str
        Path to where the test experiment is located on disk.
    config_input : str or Configuration or dict
        Path to the experiment configuration file,
        or a ``configuration_parser.Configuration`` object,
        or a dictionary with keys corresponding to fields in the
        configuration file
    suppress_warnings_for : list, optional
        Categories for which warnings should be suppressed when running the
        experiments.
        Defaults to ``[]``.
    """
    source_output_dir = "test_outputs"

    # The `csv` file extension is ultimately dropped by the `rsmpredict.py`
    # script, so these arguments can be used for CSV, TSV, or XLSX output
    output_file = join(source_output_dir, source, "output", "predictions.csv")
    feats_file = join(source_output_dir, source, "output", "preprocessed_features.csv")

    # remove all previously created files
    files = glob(join(source_output_dir, "output", "*"))
    for f in files:
        remove(f)

    with warnings.catch_warnings():
        # always suppress runtime warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # suppress additional warning types if specified
        for warning_type in suppress_warnings_for:
            warnings.filterwarnings("ignore", category=warning_type)

        compute_and_save_predictions(config_input, output_file, feats_file)


def do_run_comparison(source, config_input, suppress_warnings_for=[]):
    """
    Run rsmcompre experiment automatically.

    Use the given experiment configuration file located in the given
    source directory.

    Parameters
    ----------
    source : str
        Path to where the test experiment is located on disk.
    config_input : str or Configuration or dict
        Path to the experiment configuration file,
        or a ``configuration_parser.Configuration`` object,
        or a dictionary with keys corresponding to fields in the
        configuration file
    suppress_warnings_for : list, optional
        Categories for which warnings should be suppressed
        when running the experiments. Note that ``RuntimeWarning``s
        are always suppressed.
        Defaults to ``[]``.
    """
    source_output_dir = "test_outputs"
    experiment_dir = join(source_output_dir, source)

    with warnings.catch_warnings():
        # always suppress runtime warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # suppress additional warning types if specified
        for warning_type in suppress_warnings_for:
            warnings.filterwarnings("ignore", category=warning_type)

        run_comparison(config_input, experiment_dir)


def do_run_summary(source, config_input, suppress_warnings_for=[]):
    """
    Run rsmsummarize experiment automatically.

    Use the given experiment configuration file located in the given
    source directory.

    Parameters
    ----------
    source : str
        Path to where the test experiment is located on disk.
    config_input : str or Configuration or dict
        Path to the experiment configuration file,
        or a ``configuration_parser.Configuration`` object,
        or a dictionary with keys corresponding to fields in the
        configuration file
    suppress_warnings_for : list, optional
        Categories for which warnings should be suppressed when running the
        experiments.
        Defaults to ``[]``.
    """
    source_output_dir = "test_outputs"
    experiment_dir = join(source_output_dir, source)

    # remove all previously created files
    for output_subdir in ["output", "figure", "report"]:
        files = glob(join(source_output_dir, source, output_subdir, "*"))
        for f in files:
            remove(f)

    with warnings.catch_warnings():
        # always suppress runtime warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # suppress additional warning types if specified
        for warning_type in suppress_warnings_for:
            warnings.filterwarnings("ignore", category=warning_type)

        run_summary(config_input, experiment_dir)


def do_run_cross_validation(source, experiment_id, config_input, suppress_warnings_for=[]):
    """
    Run rsmxval experiment automatically.

    Use the given experiment configuration file located in the
    given source directory and use the given experiment ID.

    Parameters
    ----------
    source : str
        Path to where the test experiment is located on disk.
    experiment_id : str
        Experiment ID to use when running.
    config_input : str or Configuration or dict
        Path to the experiment configuration file,
        or a ``configuration_parser.Configuration`` object
        or a dictionary with keys corresponding to fields in the
        configuration file.
    suppress_warnings_for : list, optional
        Categories for which warnings should be suppressed
        when running the experiments. Note that ``RuntimeWarning``s
        are always suppressed.
        Defaults to ``[]``.
    """
    source_output_dir = "test_outputs"
    experiment_dir = join(source_output_dir, source)

    # remove all previously created files
    for output_subdir in ["folds", "fold-summary", "evaluation", "final-model"]:
        try:
            rmtree(join(source_output_dir, source, output_subdir))
        except FileNotFoundError:
            pass
    try:
        remove(join(source_output_dir, source, "rsmxval.json"))
    except FileNotFoundError:
        pass

    with warnings.catch_warnings():
        # always suppress runtime warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # suppress additional warning types if specified
        for warning_type in suppress_warnings_for:
            warnings.filterwarnings("ignore", category=warning_type)

        # call rsmxval but make sure to silence the progress bar
        # that is displayed for the parallel rsmtool runs
        run_cross_validation(config_input, experiment_dir, silence_tqdm=True)


def check_file_output(file1, file2, file_format="csv"):
    """
    Check if the two given tabular files contain matching values.

    This function checks if two experiment files have values that are
    the same to within 3 decimal places. It raises an AssertionError if
    they are not.

    Parameters
    ----------
    file1 : str
        Path to the first file.
    file2 : str
        Path to the second files.
    file_format : str, optional
        The format of the output files.
        Defaults to "csv".
    """
    # make sure that the main id columns are read as strings since
    # this may affect merging in custom notebooks
    string_columns = ["spkitemid", "candidate"]

    converter_dict = {column: str for column in string_columns}

    df1 = DataReader.read_from_file(file1, converters=converter_dict)
    df2 = DataReader.read_from_file(file2, converters=converter_dict)

    # convert all column names to strings
    # we do this to avoid any errors during sorting.
    for df in [df1, df2]:
        df.columns = df.columns.map(str)

    # if the first column is numeric, just force the index to string;
    # however, if it is non-numeric, assume that it is an index and
    # force it to string. We do this to ensure string indices are
    # preserved as such
    for df in [df1, df2]:
        if np.issubdtype(df[df.columns[0]].dtype, np.number):
            df.index = df.index.map(str)
        else:
            df.index = df[df.columns[0]]
            df.index = df.index.map(str)

    # sort all the indices alphabetically
    df1.sort_index(inplace=True)
    df2.sort_index(inplace=True)

    # sort all columns alphabetically
    df1.sort_index(axis=1, inplace=True)
    df2.sort_index(axis=1, inplace=True)

    # convert any integer columns to floats in either data frame
    for df in [df1, df2]:
        for c in df.columns:
            if df[c].dtype == np.int64:
                df[c] = df[c].astype(np.float64)

    # for pca and factor correlations convert all values to absolutes
    # because the sign may not always be the same
    if file1.endswith(f"pca.{file_format}") or file1.endswith(f"factor_correlations.{file_format}"):
        for df in [df1, df2]:
            msk = df.dtypes == np.float64
            df.loc[:, msk] = df.loc[:, msk].abs()

    try:
        assert_frame_equal(df1, df2, check_exact=False, rtol=1e-03)
    except AssertionError as e:
        message = e.args[0]
        new_message = f"File {basename(file1)} - {message}"
        e.args = (new_message,)
        raise


def collect_warning_messages_from_report(html_file):
    """
    Collect all warning messages from the given HTML report.

    Parameters
    ----------
    html_file : str
        Path to the HTML report file on disk.

    Returns
    -------
    warnings_text : list of str
        The list of collected warnings.
    """
    with open(html_file, "r") as htmlf:
        soup = BeautifulSoup(htmlf.read(), "html.parser")

    warnings_text = []
    for div in soup.findAll("div", {"class": "output_stderr"}):
        # we collect the text in the <pre> tags after the standard error,
        # and split the lines; we only keep the lines that contain 'Warning:'
        for pre in div.findAll("pre"):
            warnings_msgs = pre.text.splitlines()
            warnings_msgs = [msg for msg in warnings_msgs if "warning" in msg]
            warnings_text.extend(warnings_msgs)

    return warnings_text


def check_report(html_file, raise_errors=True, raise_warnings=True):
    """
    Raise ``AssertionError`` if given HTML report contains errors or warnings.

    Parameters
    ----------
    html_file : str
        Path to the HTML report file on disk.
    raise_errors : bool, optional
        Whether to raise an ``AssertionError`` if there
        are any errors in the report.
        Defaults to ``True``.
    raise_warnings : bool, optional
        Whether to raise an ``AssertionError`` if there
        are any warnings in the report.
        Defaults to ``True``.
    """
    report_errors = 0
    report_warnings = 0

    # Setting raise_warnings to false if not in STRICT mode
    if IGNORE_WARNINGS:
        raise_warnings = False

    with open(html_file, "r") as htmlf:
        for line in htmlf:
            m_error = html_error_regexp.search(line)
            if m_error:
                report_errors += 1
            m_warning = html_warning_regexp.search(line)
            if m_warning:
                # actual text of warning is in the next line of HTML file
                warning_text = htmlf.readline()

                # NOTE: there is a separate function
                # ``collect_warning_messages_from_the_report`` that once again
                # checks for warnings. The warnings filtered here might still
                # be flagged by that function.
                # See https://github.com/EducationalTestingService/rsmtool/issues/539

                # we do not want to flag matlplotlib font cache warning
                if not re.search(r"font\s*cache", warning_text, flags=re.IGNORECASE):
                    report_warnings += 1

    if raise_errors:
        assert report_errors == 0

    if raise_warnings:
        assert report_warnings == 0


def check_scaled_coefficients(output_dir, experiment_id, file_format="csv"):
    """
    Check that predictions using scaled coefficients match scaled scores.

    Parameters
    ----------
    output_dir : str
         Path to the experiment output directory for a test.
    experiment_id : str
        The experiment ID.
    file_format : str, optional
        The format of the output files.
        Defaults to "csv".
    """
    preprocessed_test_file = join(
        output_dir, f"{experiment_id}_test_preprocessed_features.{file_format}"
    )
    scaled_coefficients_file = join(
        output_dir, f"{experiment_id}_coefficients_scaled.{file_format}"
    )
    predictions_file = join(output_dir, f"{experiment_id}_pred_processed.{file_format}")

    df_preprocessed_test_data = DataReader.read_from_file(preprocessed_test_file)
    df_old_predictions = DataReader.read_from_file(predictions_file)
    df_old_predictions = df_old_predictions[["spkitemid", "sc1", "scale"]]

    # create fake skll objects with new coefficients
    df_coef = DataReader.read_from_file(scaled_coefficients_file)
    learner = Modeler().create_fake_skll_learner(df_coef)
    modeler = Modeler.load_from_learner(learner)

    # generate new predictions and rename the prediction column to 'scale'
    df_new_predictions = modeler.predict(df_preprocessed_test_data)
    df_new_predictions.rename(columns={"raw": "scale"}, inplace=True)

    # check that new predictions match the scaled old predictions
    assert_frame_equal(
        df_new_predictions.sort_index(axis=1),
        df_old_predictions.sort_index(axis=1),
        check_exact=False,
        rtol=1e-03,
    )


def check_generated_output(generated_files, experiment_id, model_source, file_format="csv"):
    """
    Check that all necessary output files have been generated.

    Parameters
    ----------
    generated_files : list of str
        List of files generated by a test.
    experiment_id : str
        The experiment ID.
    model_source : str
        One of "rsmtool" or "skll".
    file_format : str, optional
        The format of the output files.
        Defaults to "csv".
    """
    file_must_have_both = [
        f"_confMatrix.{file_format}",
        f"_cors_orig.{file_format}",
        f"_cors_processed.{file_format}",
        f"_eval.{file_format}",
        f"_eval_short.{file_format}",
        f"_feature.{file_format}",
        f"_feature_descriptives.{file_format}",
        f"_feature_descriptivesExtra.{file_format}",
        f"_feature_outliers.{file_format}",
        f"_margcor_score_all_data.{file_format}",
        f"_pca.{file_format}",
        f"_pcavar.{file_format}",
        f"_pcor_score_all_data.{file_format}",
        f"_pred_processed.{file_format}",
        f"_pred_train.{file_format}",
        f"_score_dist.{file_format}",
        f"_train_preprocessed_features.{file_format}",
        f"_test_preprocessed_features.{file_format}",
        f"_postprocessing_params.{file_format}",
    ]

    file_must_have_rsmtool = [f"_betas.{file_format}", f"_coefficients.{file_format}"]
    if model_source == "rsmtool":
        file_must_have = file_must_have_both + file_must_have_rsmtool
    else:
        file_must_have = file_must_have_both

    file_must_with_id = [experiment_id + file_name for file_name in file_must_have]
    file_exist = [basename(file_name) for file_name in generated_files]
    missing_file = set(file_must_with_id).difference(set(file_exist))
    assert len(missing_file) == 0, f"Missing files: {','.join(missing_file)}"


def check_consistency_files_exist(generated_files, experiment_id, file_format="csv"):
    """
    Check that the consistency files were generated.

    Parameters
    ----------
    generated_files : list of str
        List of files generated by a test.
    experiment_id : str
        The experiment ID.
    file_format : str, optional
        The format of the output files.
        Defaults to "csv".
    """
    file_must_have = [
        f"_consistency.{file_format}",
        f"_degradation.{file_format}",
        f"_disattenuated_correlations.{file_format}",
        f"_true_score_eval.{file_format}",
    ]

    file_must_with_id = [experiment_id + file_name for file_name in file_must_have]
    file_exist = [basename(file_name) for file_name in generated_files]
    missing_file = set(file_must_with_id).difference(set(file_exist))
    assert len(missing_file) == 0, f"Missing files: {','.join(missing_file)}"


def check_subgroup_outputs(output_dir, experiment_id, subgroups, file_format="csv"):
    """
    Check that the subgroup-related outputs are accurate.

    Parameters
    ----------
    output_dir : str
        Path to the `output` experiment output directory for a test.
    experiment_id : str
        The experiment ID.
    subgroups : list of str
        List of column names that contain grouping
        information.
    file_format : str, optional
        The format of the output files.
        Defaults to "csv".
    """
    train_preprocessed_file = join(output_dir, f"{experiment_id}_train_metadata.{file_format}")
    train_preprocessed = DataReader.read_from_file(train_preprocessed_file, index_col=0)

    test_preprocessed_file = join(output_dir, f"{experiment_id}_test_metadata.{file_format}")
    test_preprocessed = DataReader.read_from_file(test_preprocessed_file, index_col=0)
    for group in subgroups:
        assert group in train_preprocessed.columns
        assert group in test_preprocessed.columns

    # check that the total sum of N per category matches the total N
    # in data composition and the total N categories matches what is
    # in overall data composition
    file_data_composition_all = join(output_dir, f"{experiment_id}_data_composition.{file_format}")
    df_data_composition_all = DataReader.read_from_file(file_data_composition_all)
    for group in subgroups:
        file_composition_by_group = join(
            output_dir, f"{experiment_id}_data_composition_by_{group}.{file_format}"
        )
        composition_by_group = DataReader.read_from_file(file_composition_by_group)
        for partition in ["Training", "Evaluation"]:
            partition_info = df_data_composition_all.loc[
                df_data_composition_all["partition"] == partition
            ]

            summation = sum(composition_by_group[f"{partition} set"])
            assert summation == partition_info.iloc[0]["responses"]

            length = len(composition_by_group.loc[composition_by_group[f"{partition} set"] != 0])
            assert length == partition_info.iloc[0][group]


def copy_data_files(temp_dir_name, input_file_dict, given_test_dir):
    """
    Copy files from given test directory to a temporary directory.

    Useful for tests where the current directory is to be used as the
    reference for resolving paths in the configuration.

    Parameters
    ----------
    temp_dir_name : str
        Name of the temporary directory.
    input_file_dict : dict
        A dictionary of files/directories to copy with keys as the
        file type and the values are their paths relative to the ``tests``
        directory.
    given_test_dir : str
        Directory where the the test experiments are located. This can be
        useful when using these experiments to run tests for RSMExtra.

    Returns
    -------
    output_file_dict : dict
        The dictionary with the same keys as
        ``input_file_dict`` and values being the copied paths.
    """
    temp_dir = Path(temp_dir_name)
    if not temp_dir.exists():
        temp_dir.mkdir()

    output_file_dict = {}
    for file in input_file_dict:
        filepath = Path(input_file_dict[file])
        filename = filepath.name
        old_filepath = given_test_dir / filepath
        new_filepath = temp_dir / filename
        if old_filepath.is_dir():
            copytree(old_filepath, new_filepath)
        else:
            copyfile(old_filepath, new_filepath)
        output_file_dict[file] = str(new_filepath)

    return output_file_dict


class FileUpdater(object):
    """
    Class used to update outputs for tests.

    A FileUpdater object is used to update the test outputs
    for the tests in the ``tests_directory`` based on the
    outputs contained in the ``updated_outputs_directory``.
    It does this for all of the experiment tests contained
    in the test files given by each of the ``test_suffixes``.

    Attributes
    ----------
    test_suffixes : list
        List of suffixes that will be added to the string
        "test_experiment_" and located in the ``tests_directory``
        to find the tests that are to be updated.
    tests_directory : str
        Path to the directory containing the tests whose outputs are
        to be updated.
    updated_outputs_directory : str
        Path to the directory containing the updated outputs for the
        experiment tests.
    deleted_files : list
        List of files deleted from ``tests directory``.
    updated_files : list
        List of files that have either (really) changed in the updated outputs
        or been added in those outputs.
    missing_or_empty_sources : list
        List of source names whose corresponding directories are either
        missing under ``updated_outputs_directory` or do exist but are
        empty.
    """

    def __init__(self, test_suffixes, tests_directory, updated_outputs_directory):
        """Instantiate a FileUpdater object."""
        self.test_suffixes = test_suffixes
        self.tests_directory = Path(tests_directory)
        self.updated_outputs_directory = Path(updated_outputs_directory)
        self.missing_or_empty_sources = []
        self.deleted_files = []
        self.updated_files = []

        # invalidate the file comparison cache
        clear_cache()

    def is_skll_excluded_file(self, filename):
        """
        Check whether given filename should be excluded for SKLL-based tests.

        Parameters
        ----------
        filename : str
            Name of the file to be checked.

        Returns
        -------
        exclude : bool
            ``True`` if the file should be excluded.
            ``False`` otherwise.
        """
        possible_suffixes = [".model", ".npy"]
        possible_stems = [
            "_postprocessing_params",
            "_eval",
            "_eval_short",
            "_confMatrix",
            "_pred_train",
            "_pred_processed",
            "_score_dist",
        ]

        file_stem = Path(filename).stem
        file_suffix = Path(filename).suffix
        return any(file_suffix == suffix for suffix in possible_suffixes) or any(
            file_stem.endswith(stem) for stem in possible_stems
        )

    def update_source(self, source, skll=False, file_type="output", input_source=None):
        """
        Update test output or input data for test named ``source``.

        This method updates the test output or input data for experiment test
        with ``source`` as the given name. It deletes files that are only in
        the tests directory, adds files that are only in the updated test
        outputs directory, and updates the files that have changed in the
        updated test outputs directory. It does not return anything but
        updates the ``deleted_files``, ``updated_files``, and
        ``missing_or_empty_sources`` class attributes appropriately.

        Parameters
        ----------
        source : str
            Name of source directory.
        skll : bool, optional
            Whether the given source is for a SKLL-based test.
            Defaults to ``False``.
        file_type: str, optional
            Whether we are updating test output files or test input files.
            Input files are updated for rsmtool and rsmcompare.
            Defaults to "output".
        input_source: str, optional
            The name of the source directory for input files
            Defaults to ``None``.
        """
        # locate the updated outputs for the experiment under the given
        # outputs directory, locate the existing experiment outputs
        # and define how we will refer to the test
        all_updated_folders = []
        test_name = source
        if file_type == "output":
            # for xval we need to update data from several output folder
            if "xval" in source:
                updated_xval_path = self.updated_outputs_directory / source
                for output_sub_folder in ["evaluation", "final-model", "fold-summary"]:
                    updated_output_path = updated_xval_path / output_sub_folder / "output"
                    existing_output_path = (
                        self.tests_directory
                        / "data"
                        / "experiments"
                        / source
                        / "output"
                        / output_sub_folder
                        / "output"
                    )
                    all_updated_folders.append((updated_output_path, existing_output_path))
                updated_folds_path = updated_xval_path / "folds"
                for fold in os.listdir(updated_folds_path):
                    updated_fold_path = updated_folds_path / fold / "output"
                    existing_fold_path = (
                        self.tests_directory
                        / "data"
                        / "experiments"
                        / source
                        / "output"
                        / "folds"
                        / fold
                        / "output"
                    )
                    all_updated_folders.append((updated_fold_path, existing_fold_path))
            # for all other experiment types, we havea  single output folder
            else:
                updated_output_path = self.updated_outputs_directory / source / "output"
                existing_output_path = (
                    self.tests_directory / "data" / "experiments" / source / "output"
                )
                all_updated_folders.append((updated_output_path, existing_output_path))
        else:
            updated_output_path = self.updated_outputs_directory / input_source / "output"
            existing_output_path = (
                self.tests_directory / "data" / "experiments" / source / input_source / "output"
            )
            all_updated_folders.append((updated_output_path, existing_output_path))
            test_name += f"/{input_source}"

        for updated_output_path, existing_output_path in all_updated_folders:
            # if the directory for this source does not exist on the updated output
            # side, then that's a problem and something we should report on later
            try:
                assert updated_output_path.exists()
            except AssertionError:
                self.missing_or_empty_sources.append(test_name)
                return

            # if the existing output path does not exist, then create it
            try:
                assert existing_output_path.exists()
            except AssertionError:
                sys.stderr.write(
                    f'\nNo existing output for "{test_name}". Creating directory ...\n'
                )
                existing_output_path.mkdir(parents=True)

            # get a comparison between the two directories
            dir_comparison = dircmp(updated_output_path, existing_output_path)

            # if no output was found in the updated outputs directory, that's
            # likely to be a problem so save that source
            if not dir_comparison.left_list:
                self.missing_or_empty_sources.append(test_name)
                return

            # first delete the files that only exist in the existing output directory
            # since those are likely old files from old versions that we do not need
            existing_output_only_files = dir_comparison.right_only
            for file in existing_output_only_files:
                remove(existing_output_path / file)

            # Next find all the NEW files in the updated outputs.
            new_files = dir_comparison.left_only

            # We also define several types of files we exclude.
            # 1. we exclude OLS summary files
            excluded_suffixes = ["_ols_summary.txt", ".ols", ".model", ".npy"]

            # 2. for output files we exclude all json files.
            # We keep these files if we are dealing with input files.
            if file_type == "output":
                excluded_suffixes.extend(
                    [
                        "_rsmtool.json",
                        "_rsmeval.json",
                        "_rsmsummarize.json",
                        "_rsmcompare.json",
                        "_rsmxval.json",
                    ]
                )

            new_files = [
                f for f in new_files if not any(f.endswith(suffix) for suffix in excluded_suffixes)
            ]

            # 3. We also exclude files related to model evaluations for SKLL models.
            if skll:
                new_files = [f for f in new_files if not self.is_skll_excluded_file(f)]

            # next we get the files that have changed and try to figure out if they
            # have actually changed beyond a tolerance level that we care about for
            # tests. To do this, we run the same function that we use when comparing
            # the files in the actual test. However, for non-tabular files, we just
            # assume that they have really changed since we have no easy way to compare.
            changed_files = dir_comparison.diff_files
            really_changed_files = []
            for changed_file in changed_files:
                include_file = True
                updated_output_filepath = updated_output_path / changed_file
                existing_output_filepath = existing_output_path / changed_file
                file_format = updated_output_filepath.suffix.lstrip(".")
                if file_format in ["csv", "tsv", "xlsx"]:
                    try:
                        check_file_output(
                            str(updated_output_filepath),
                            str(existing_output_filepath),
                            file_format=file_format,
                        )
                    except AssertionError:
                        pass
                    else:
                        include_file = False

                if include_file:
                    really_changed_files.append(changed_file)

            # Copy over the new files as well as the really changed files
            new_or_changed_files = new_files + really_changed_files
            for file in new_or_changed_files:
                copyfile(updated_output_path / file, existing_output_path / file)

            # Update the lists with files that were changed for this source
            self.deleted_files.extend([(test_name, file) for file in existing_output_only_files])
            self.updated_files.extend([(test_name, file) for file in new_or_changed_files])

    def update_test_data(self, source, test_tool, skll=False):
        """
        Determine whether to update input or output data and run ``update_source()``.

        Parameters
        ----------
        source : str
            Name of source directory.
        test_tool : str
            What tool is tested by this test.
        skll : bool, optional
            Whether the given source is for a SKLL-based test.
            Defaults to ``False``.
        """
        existing_output_path = self.tests_directory / "data" / "experiments" / source / "output"
        # if we have a tool without with output
        # we update the outputs
        if test_tool in tools_with_output:
            self.update_source(source, skll=skll)
        # if we have a tool with input data we also update inputs
        if test_tool in tools_with_input_data:
            for input_dir in existing_output_path.parent.iterdir():
                if not input_dir.is_dir():
                    continue
                if input_dir.name in ["output", "figure", "report"]:
                    continue
                else:
                    input_source = input_dir.name
                    self.update_source(
                        source, skll=skll, file_type="input", input_source=input_source
                    )

    def run(self):
        """Update test data in files given by the ``test_suffixes`` attribute."""
        # import all the test_suffix experiment files using SourceFileLoader
        # adapted from: https://stackoverflow.com/a/67692
        for test_suffix in self.test_suffixes:
            test_module_path = join(self.tests_directory, f"test_experiment_{test_suffix}.py")
            test_module = SourceFileLoader(f"loaded_{test_suffix}", test_module_path).load_module()
            test_tool = test_suffix.split("_")[0]

            # skip the module if it tells us that it doesn't want the data for its tests updated
            if hasattr(test_module, "_AUTO_UPDATE"):
                if not test_module._AUTO_UPDATE:
                    continue

            # iterate over all the members and focus on only the experiment classes
            # and methods. For rsmtool/rsmeval we skip over the functions that are
            # decorated with '@raises' since those functions do not need any test
            # data to be updated. For rsmsummarize and rsmcompare we only update
            # the input files for these functions. For the rest, try to get
            # the source since that's what we need to update the test files.
            for member_name, member_object in getmembers(test_module):
                if isclass(member_object) and member_name.startswith("Test"):
                    test_class = member_object

                    # now iterate over all the test methods in the test class
                    for class_member_name, class_member_object in getmembers(test_class):
                        if isfunction(class_member_object) and (
                            class_member_name.startswith("test_run_experiment")
                            or class_member_name.startswith("test_run_cross_validation")
                        ):
                            function = class_member_object

                            # get the qualified name of the member function

                            # check if the member function uses `assertRaises`
                            source = getsource(function)
                            if "with self.assertRaises" in source:
                                continue

                            # otherwise first we check if it's the parameterized function and if so
                            # we can easily get the source from the parameter list
                            if class_member_name.endswith("parameterized"):
                                for param in function.paramList:
                                    source_name = param["source"]
                                    skll = param.get("skll", False)
                                    self.update_test_data(source_name, test_tool, skll=skll)

                            # if it's another function, then we actually inspect the code
                            # to get the source. Note that this should never be a SKLL experiment
                            # since those should always be run parameterized
                            else:
                                function_code_lines = getsourcelines(function)
                                source_line = [
                                    line
                                    for line in function_code_lines[0]
                                    if re.search(r"source = ", line)
                                ]
                                source_name = eval(source_line[0].strip().split(" = ")[1])
                                self.update_test_data(source_name, test_tool)

    def print_report(self):
        """Print a report of all changes made when the updater was run."""
        # print out the number and list of overall deleted files
        print(f"{len(self.deleted_files)} deleted:")
        for source, deleted_file in self.deleted_files:
            print(f"{source} {deleted_file}")
        print()

        # find added/updated input files: in this case the source # will consist of
        # the test name and the input test name separated by '/'.
        updated_input_files = [
            (source, updated_file) for (source, updated_file) in self.updated_files if "/" in source
        ]

        # print out the number and list of overall added/updated non-model files
        print(f"{len(self.updated_files)} added/updated:")
        for source, updated_file in self.updated_files:
            print(f"{source} {updated_file}")
        print()

        # now print out missing and/or empty updated output directories
        print(f"{len(self.missing_or_empty_sources)} missing/empty sources in updated outputs:")
        for source in self.missing_or_empty_sources:
            print(f"{source}")
        print()

        # if we updated any input files, let the user know that they need to
        # re-run the tests and update test outputs
        if len(updated_input_files) > 0:
            print(
                f"WARNING: {len(updated_input_files)} input files for rsmcompare/rsmsummarize "
                f"tests have been updated. You need to re-run these tests and update test outputs"
            )
