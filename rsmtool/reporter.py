"""
Classes for dealing with report generation.

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import warnings
from os.path import abspath, basename, dirname, join, normpath, splitext

import nbformat
from nbconvert.exporters import HTMLExporter, NotebookExporter
from nbconvert.exporters.templateexporter import default_filters
from nbformat.warnings import DuplicateCellId, MissingIDFieldWarning
from traitlets.config import Config

from . import HAS_RSMEXTRA
from .reader import DataReader
from .utils.wandb import log_report_to_wandb

if HAS_RSMEXTRA:
    from rsmextra.settings import special_section_list_rsmtool  # noqa
    from rsmextra.settings import (
        ordered_section_list_with_special_sections_rsmcompare,
        ordered_section_list_with_special_sections_rsmeval,
        ordered_section_list_with_special_sections_rsmsummarize,
        ordered_section_list_with_special_sections_rsmtool,
        special_notebook_path,
        special_section_list_rsmcompare,
        special_section_list_rsmeval,
        special_section_list_rsmsummarize,
    )

    ordered_section_list_rsmtool = ordered_section_list_with_special_sections_rsmtool
    ordered_section_list_rsmeval = ordered_section_list_with_special_sections_rsmeval
    ordered_section_list_rsmcompare = ordered_section_list_with_special_sections_rsmcompare
    ordered_section_list_rsmsummarize = ordered_section_list_with_special_sections_rsmsummarize

else:
    ordered_section_list_rsmtool = [
        "data_description",
        "data_description_by_group",
        "feature_descriptives",
        "features_by_group",
        "preprocessed_features",
        "dff_by_group",
        "consistency",
        "model",
        "evaluation",
        "true_score_evaluation",
        "evaluation_by_group",
        "fairness_analyses",
        "pca",
        "intermediate_file_paths",
        "sysinfo",
    ]

    ordered_section_list_rsmeval = [
        "data_description",
        "data_description_by_group",
        "consistency",
        "evaluation",
        "true_score_evaluation",
        "evaluation_by_group",
        "fairness_analyses",
        "intermediate_file_paths",
        "sysinfo",
    ]

    ordered_section_list_rsmcompare = [
        "feature_descriptives",
        "features_by_group",
        "preprocessed_features",
        "preprocessed_features_by_group",
        "consistency",
        "score_distributions",
        "model",
        "evaluation",
        "true_score_evaluation",
        "pca",
        "notes",
        "sysinfo",
    ]

    ordered_section_list_rsmsummarize = [
        "preprocessed_features",
        "model",
        "evaluation",
        "true_score_evaluation",
        "intermediate_file_paths",
        "sysinfo",
    ]

    ordered_section_list_rsmexplain = ["data_description", "shap_values", "shap_plots", "sysinfo"]

    special_section_list_rsmtool = []
    special_section_list_rsmcompare = []
    special_section_list_rsmeval = []
    special_section_list_rsmsummarize = []
    special_section_list_rsmexplain = []
    special_notebook_path = ""

package_path = dirname(__file__)
notebook_path = abspath(join(package_path, "notebooks"))
template_path = join(notebook_path, "templates")

javascript_path = join(notebook_path, "javascript")
comparison_notebook_path = join(notebook_path, "comparison")
summary_notebook_path = join(notebook_path, "summary")
explanations_notebook_path = join(notebook_path, "explanations")

# Define the general section list

general_section_list_rsmtool = [
    section
    for section in ordered_section_list_rsmtool
    if section not in special_section_list_rsmtool
]

general_section_list_rsmeval = [
    section
    for section in ordered_section_list_rsmeval
    if section not in special_section_list_rsmeval
]

general_section_list_rsmcompare = [
    section
    for section in ordered_section_list_rsmcompare
    if section not in special_section_list_rsmcompare
]

general_section_list_rsmsummarize = [
    section
    for section in ordered_section_list_rsmsummarize
    if section not in special_section_list_rsmsummarize
]

general_section_list_rsmexplain = [
    section
    for section in ordered_section_list_rsmexplain
    if section not in special_section_list_rsmexplain
]

# define a mapping from the tool name to the master
# list for both general and special sections
master_section_dict = {
    "general": {
        "rsmtool": general_section_list_rsmtool,
        "rsmeval": general_section_list_rsmeval,
        "rsmcompare": general_section_list_rsmcompare,
        "rsmsummarize": general_section_list_rsmsummarize,
        "rsmexplain": general_section_list_rsmexplain,
    },
    "special": {
        "rsmtool": special_section_list_rsmtool,
        "rsmeval": special_section_list_rsmeval,
        "rsmcompare": special_section_list_rsmcompare,
        "rsmsummarize": special_section_list_rsmsummarize,
        "rsmexplain": special_section_list_rsmexplain,
    },
}

# define the mapping for section paths
notebook_path_dict = {
    "general": {
        "rsmtool": notebook_path,
        "rsmeval": notebook_path,
        "rsmcompare": comparison_notebook_path,
        "rsmsummarize": summary_notebook_path,
        "rsmexplain": explanations_notebook_path,
    },
    "special": {
        "rsmtool": special_notebook_path,
        "rsmeval": special_notebook_path,
        "rsmcompare": special_notebook_path,
        "rsmsummarize": special_notebook_path,
    },
}


class Reporter:
    """Class to generate Jupyter notebook reports and convert them to HTML."""

    def __init__(self, logger=None, wandb_run=None):
        """Initialize the Reporter object."""
        self.logger = logger if logger else logging.getLogger(__name__)
        self.wandb_run = wandb_run

    @staticmethod
    def locate_custom_sections(custom_report_section_paths, configdir):
        """
        Locate custom report section files.

        Get the absolute paths for custom report sections and check that
        the files exist. If a file does not exist, raise an exception.

        Parameters
        ----------
        custom_report_section_paths : list of str
            List of paths to IPython notebook
            files representing the custom sections.
        configdir : str
            Path to the experiment configuration directory.

        Returns
        -------
        custom_report_sections :  list of str
            List of absolute paths to the custom section
            notebooks.

        Raises
        ------
        FileNotFoundError
            If any of the files cannot be found.
        """
        custom_report_sections = []
        for cs_path in custom_report_section_paths:
            cs_location = DataReader.locate_files(cs_path, configdir)
            if not cs_location:
                raise FileNotFoundError(f"Error: custom section not found at {cs_path}.")
            else:
                custom_report_sections.append(cs_location)
        return custom_report_sections

    @staticmethod
    def merge_notebooks(notebook_files, output_file):
        """
        Merge the given Jupyter notebooks into a single Jupyter notebook.

        Parameters
        ----------
        notebook_files : list of str
            List of paths to the input Jupyter notebook files.
        output_file : str
            Path to output Jupyter notebook file
        """
        # create a new blank notebook
        merged_notebook = nbformat.v4.new_notebook()

        # append the cells from all of the notebooks to this notebook
        for nbfile in notebook_files:
            with open(nbfile, "r") as nbfh:
                nb = nbformat.read(nbfh, as_version=4)

            for cell in nb.cells:
                merged_notebook.cells.append(cell)

        # normalize the merged notebook to fix any issues with metadata
        # especially missing or duplicate IDs in code cells which seems to
        # happen a lot for our sections
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=MissingIDFieldWarning)
            warnings.filterwarnings("ignore", category=DuplicateCellId)
            _, merged_notebook = nbformat.validator.normalize(merged_notebook.dict())

        # write out the merged notebook to disk
        exporter = NotebookExporter()
        output, _ = exporter.from_notebook_node(merged_notebook)
        with open(output_file, "w") as outfh:
            outfh.write(output)

    @staticmethod
    def check_section_names(specified_sections, section_type, context="rsmtool"):
        """
        Validate the specified section names.

        This function checks whether the specified section names are valid
        and raises an exception if they are not.

        Parameters
        ----------
        specified_sections : list of str
            List of report section names.
        section_type : str
            One of "general" or "special".
        context : str, optional
            Context in which we are validating the section
            names. One of {"rsmtool", "rsmeval", "rsmcompare"}.
            Defaults to "rsmtool".

        Raises
        ------
        ValueError
            If any of the section names of the given type
            are not valid in the context of the given tool.
        """
        master_section_list = master_section_dict[section_type][context]
        invalid_section_names = set(specified_sections).difference(master_section_list)
        if invalid_section_names:
            raise ValueError(
                f"The following {section_type} report section names are "
                f"invalid or not supported for {context}: "
                f"{invalid_section_names}\nThe following sections are "
                f"currently available: {master_section_list}"
            )

    @staticmethod
    def check_section_order(chosen_sections, section_order):
        """
        Check the order of the specified sections.

        Parameters
        ----------
        chosen_sections : list of str
            List of chosen section names.
        section_order : list of str
            An ordered list of the chosen section names.

        Raises
        ------
        ValueError
            If any sections specified in the order are missing
            from the list of chosen sections or vice versa.
        """
        if sorted(chosen_sections) != sorted(section_order):
            # check for discrepancies and create a helpful error message
            missing_sections = set(chosen_sections).difference(set(section_order))
            if missing_sections:
                error_message_missing = (
                    f"'section_order' must list all sections "
                    f"selected for your experiment: Please edit "
                    f"section order to include the following missing "
                    f"sections: {', '.join(missing_sections)}"
                )

            extra_sections = set(section_order).difference(set(chosen_sections))
            if extra_sections:
                error_message_extra = (
                    f"'section order' can only include sections "
                    f"available for this experiment. The following "
                    f"sections are either unavailable or were not "
                    f"selected for this experiment {', '.join(extra_sections)}"
                )

            # raise an appropriate error message or a combination of messages
            if missing_sections and not extra_sections:
                raise ValueError(error_message_missing)
            elif extra_sections and not missing_sections:
                raise ValueError(error_message_extra)
            else:
                raise ValueError(f"{error_message_missing}\n{error_message_extra}")

    @staticmethod
    def convert_ipynb_to_html(notebook_file, html_file):
        """
        Convert given Jupyter notebook (``.ipynb``) to HTML file.

        Parameters
        ----------
        notebook_file : str
            Path to input Jupyter notebook file.
        html_file : str
            Path to output HTML file.

        Note
        ----
        This function is also exposed as the
        :ref:`render_notebook <render_notebook>` command-line utility.
        """
        # `nbconvert` uses `asyncio` which uses an entirely default
        # implemention of the event loop on Windows for Cpython 3.8
        # which breaks the report generation unless we include the
        # following workaround
        if sys.version_info[0] == 3 and sys.version_info[1] >= 8 and sys.platform.startswith("win"):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

        # set this environment variable for Python 3.11 since otherwise
        # invoking anything ipython-related yields warnings about debugging
        # frozen modules
        if sys.version_info[0] == 3 and sys.version_info[1] >= 11:
            os.environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"

        # set a high timeout for datasets with a large number of features
        report_config = Config(
            {
                "ExecutePreprocessor": {"enabled": True, "timeout": 3600},
                "HTMLExporter": {
                    "template_name": "classic",
                    "template_file": join(template_path, "report.tpl"),
                },
            }
        )

        # newer versions of nbconvert use a "clean_html" filter that
        # break SVG rendering; so we override that filter here with
        # a custom function that is essentially a noop. For more
        # details, please refer to the following issue:
        # https://github.com/EducationalTestingService/rsmtool/issues/571
        def custom_clean_html(element):
            return element.decode() if isinstance(element, bytes) else str(element)

        default_filters["clean_html"] = custom_clean_html

        # we want to suppress the logged warning from nbconvert about missing
        # alt text since there is currently no easy way to add alt text to
        # inlined-SVG images that are generated programmatically. To do this,
        # we will temporarily change the log level for the log object that
        # produces this warning and then restore it after we are done
        exportHtml = HTMLExporter(config=report_config)
        old_level = exportHtml.log.level
        exportHtml.log.setLevel(logging.ERROR)
        output, _ = exportHtml.from_filename(notebook_file)
        exportHtml.log.setLevel(old_level)
        open(html_file, mode="w", encoding="utf-8").write(output)

    def determine_chosen_sections(
        self,
        general_sections,
        special_sections,
        custom_sections,
        subgroups,
        context="rsmtool",
    ):
        """
        Compile a combined list of section names to be included in the report.

        Parameters
        ----------
        general_sections : list of str
            List of specified general section names.
        special_sections : str
            List of specified special section names, if any.
        custom_sections : list of str
            List of specified custom sections, if any.
        subgroups : list of str
            List of column names that contain grouping
            information.
        context : str, optional
            Context of the tool in which we are validating.
            One of  {"rsmtool", "rsmeval", "rsmcompare"}
            Defaults to "rsmtool".

        Returns
        -------
        chosen_sections : list of str
            Final list of chosen sections that are to
            be included in the HTML report.

        Raises
        ------
        ValueError
            If a subgroup report section is requested but no
            subgroups were specified in the configuration file.
        """
        # 1. Include all general sections unless we are asked to include
        # a specific (and valid) subset.
        general_section_list = master_section_dict["general"][context]
        chosen_general_sections = general_section_list
        all_general_sections = True

        if general_sections != ["all"]:
            self.check_section_names(general_sections, "general", context)
            chosen_general_sections = [s for s in general_sections if s in general_section_list]
            all_general_sections = False

        # 2. Exclude the subgroup sections if we do not have subgroup information.

        if len(subgroups) == 0:
            subgroup_sections = [
                section
                for section in chosen_general_sections
                if section.endswith("by_group") or section == "fairness_analyses"
            ]
            # if we were given a list of general sections, raise an error if
            # that list included subgroup sections but no subgroups were specified

            if not all_general_sections and len(subgroup_sections) != 0:
                raise ValueError(
                    f"You requested sections for subgroup analysis "
                    f"but did not specify any subgroups. Please amend "
                    f"the config files to define the subgroups or delete "
                    f"the following sections from the list of sections: "
                    f"{', '.join(subgroup_sections)}"
                )

            # if we are using the default list, we simply remove the
            # subgroup sections
            chosen_general_sections = [
                section for section in chosen_general_sections if section not in subgroup_sections
            ]

        # 3. Include the specified (and valid) subset of the special sections
        chosen_special_sections = []
        if special_sections:
            special_section_list = master_section_dict["special"][context]
            self.check_section_names(special_sections, "special", context=context)
            chosen_special_sections = [s for s in special_sections if s in special_section_list]

        # 4. For the custom sections use the basename and strip off the `.ipynb` extension
        chosen_custom_sections = []
        if custom_sections:
            chosen_custom_sections = [splitext(basename(cs))[0] for cs in custom_sections]

        # return the final list of chosen sections
        chosen_sections = chosen_general_sections + chosen_special_sections + chosen_custom_sections

        return chosen_sections

    def get_section_file_map(
        self, special_sections, custom_sections, model_type=None, context="rsmtool"
    ):
        """
        Map section names to IPython notebook filenames.

        Parameters
        ----------
        special_sections : list of str
            List of special sections.
        custom_sections : list of str
            List of custom sections.
        model_type : str, optional
            Type of the model. One of {"BUILTIN", "SKLL", ``None``}.
            We allow ``None`` here so that rsmeval can use the same
            function.
            Defaults to ``None``.
        context : str, optional
            Context of the tool in which we are validating.
            One of {"rsmtool", "rsmeval", "rsmcompare"}.
            Defaults to "rsmtool".

        Returns
        -------
        section_file_map : dict
            Dictionary mapping each section name to the
            corresponding IPython notebook filename.
        """
        # create the original section file map for general sections
        selected_notebook_path = notebook_path_dict["general"][context]
        general_sections = master_section_dict["general"][context]

        section_file_map = {
            s: join(selected_notebook_path, f"{s}.ipynb")
            for s in general_sections + ["header", "footer"]
        }

        # update the file map to point the 'model section to either the built-in
        # or the SKLL model notebook depending on the model type that
        # was passed in
        if context == "rsmtool" and model_type:
            section_file_map["model"] = join(
                selected_notebook_path, f"{model_type.lower()}_model.ipynb"
            )

        # update the file map to include the special sections
        if special_sections:
            selected_special_notebook_path = notebook_path_dict["special"][context]
            section_file_map.update(
                {ss: join(selected_special_notebook_path, f"{ss}.ipynb") for ss in special_sections}
            )

        # update the file map to include the custom sections with
        # the file names (without the `.ipynb` extension) as the
        # names (keys) and full paths as values
        if custom_sections:
            section_file_map.update({splitext(basename(cs))[0]: cs for cs in custom_sections})

        return section_file_map

    def get_ordered_notebook_files(
        self,
        general_sections,
        special_sections=[],
        custom_sections=[],
        section_order=None,
        subgroups=[],
        model_type=None,
        context="rsmtool",
    ):
        """
        Check all section names and the order of the sections.

        Combine all section names with the appropriate file mapping,
        and generate an ordered list of notebook files that are
        needed to generate the final report.

        Parameters
        ----------
        general_sections : str
            List of specified general sections.
        special_sections : list, optional
            List of specified special sections, if any.
            Defaults to ``[]``.
        custom_sections : list, optional
            List of specified custom sections, if any.
            Defaults to ``[]``.
        section_order : list, optional
            Ordered list in which the user wants the specified
            sections.
            Defaults to ``None``.
        subgroups : list, optional
            List of column names that contain grouping
            information.
            Defaults to ``[]``.
        model_type : None, optional
            Type of the model. Possible values are
            {"BUILTIN", "SKLL", ``None``.}. We allow ``None``
            here so that rsmeval can use the same function.
            Defaults to ``None``.
        context : str, optional
            Context of the tool in which we are validating.
            One of {"rsmtool", "rsmeval", "rsmcompare"}.
            Defaults to "rsmtool".

        Returns
        -------
        chosen_notebook_files : list of str
            List of the IPython notebook files that have
            to be rendered into the HTML report.
        """
        chosen_sections = self.determine_chosen_sections(
            general_sections,
            special_sections,
            custom_sections,
            subgroups,
            context=context,
        )

        # check to make sure that if a custom section ordering is
        # specified by the user, that it actually contains
        # *all* of the sections that have been chosen for the
        # final report.
        if section_order:
            self.check_section_order(chosen_sections, section_order)

        # determine which order to use by default
        if context == "rsmtool":
            ordered_section_list = ordered_section_list_rsmtool
        elif context == "rsmeval":
            ordered_section_list = ordered_section_list_rsmeval
        elif context == "rsmcompare":
            ordered_section_list = ordered_section_list_rsmcompare
        elif context == "rsmsummarize":
            ordered_section_list = ordered_section_list_rsmsummarize
        elif context == "rsmexplain":
            ordered_section_list = ordered_section_list_rsmexplain

        # add all custom sections to the end of the default ordered list
        ordered_section_list.extend([splitext(basename(cs))[0] for cs in custom_sections])

        # get the section file map
        section_file_map = self.get_section_file_map(
            special_sections, custom_sections, model_type, context=context
        )

        # order the section list either according to the default
        # order in `ordered_section_list` or according to the custom
        # order that has been passed in via `section_order`
        order_to_use = section_order if section_order else ordered_section_list
        chosen_sections = [s for s in order_to_use if s in chosen_sections]

        # add the header and the footer to the chosen sections
        chosen_sections = ["header"] + chosen_sections + ["footer"]
        chosen_notebook_files = [section_file_map[cs] for cs in chosen_sections]

        return chosen_notebook_files

    def create_report(self, config, csvdir, figdir, context="rsmtool"):
        """
        Generate HTML report for an rsmtool/rsmeval experiment.

        Parameters
        ----------
        config : configuration_parser.Configuration
            A configuration object
        csvdir : str
            The CSV output directory.
        figdir : str
            The figure output directory
        context : str
            Context of the tool in which we are validating.
            One of {"rsmtool", "rsmeval"}.
            Defaults to "rsmtool".

        Raises
        ------
        KeyError
            If "test_file_location" or "pred_file_location" fields
            are not specified in the configuration.
        """
        test_file_location = config.get("test_file_location")
        if test_file_location is None:
            test_file_location = config.get("pred_file_location")

        # raise error if location is still None
        if test_file_location is None:
            raise KeyError(
                "Could not find `test_file_location` or `pred_file_location` "
                "in Configuration object. Please make sure you have included "
                "one of these parameters in the configuration object."
            )

        # if the features subset file is None, just use empty string
        feature_subset_file = (
            "" if config.get("feature_subset_file") is None else config["feature_subset_file"]
        )

        # if the min items is None, just set it to zero
        min_items = (
            0 if config["min_items_per_candidate"] is None else config["min_items_per_candidate"]
        )

        # determine minimum and maximum scores for trimming
        (
            used_trim_min,
            used_trim_max,
            trim_tolerance,
        ) = config.get_trim_min_max_tolerance()
        min_score = used_trim_min - trim_tolerance
        max_score = used_trim_max + trim_tolerance

        environ_config = {
            "EXPERIMENT_ID": config["experiment_id"],
            "DESCRIPTION": config["description"],
            "MODEL_TYPE": config.get("model_type", ""),
            "MODEL_NAME": config.get("model_name", ""),
            "TRAIN_FILE_LOCATION": config.get("train_file_location", ""),
            "TEST_FILE_LOCATION": test_file_location,
            "SUBGROUPS": config.get("subgroups", []),
            "GROUPS_FOR_DESCRIPTIVES": config.get("subgroups", []),
            "GROUPS_FOR_EVALUATIONS": config.get("subgroups", []),
            "MIN_N_PER_GROUP": config.get("min_n_per_group", {}),
            "LENGTH_COLUMN": config.get("length_column", None),
            "H2_COLUMN": config["second_human_score_column"],
            "MIN_ITEMS": min_items,
            "FEATURE_SUBSET_FILE": feature_subset_file,
            "EXCLUDE_ZEROS": config.get("exclude_zero_scores", True),
            "SCALED": config.get("use_scaled_predictions", False),
            "MIN_SCORE": min_score,
            "MAX_SCORE": max_score,
            "STANDARDIZE_FEATURES": config.get("standardize_features", True),
            "FILE_FORMAT": config.get("file_format", "csv"),
            "USE_THUMBNAILS": config.get("use_thumbnails", False),
            "SKLL_FIXED_PARAMETERS": config.get("skll_fixed_parameters", {}),
            "SKLL_OBJECTIVE": config.get("skll_objective", ""),
            "PREDICT_EXPECTED_SCORES": config.get("predict_expected_scores", False),
            "RATER_ERROR_VARIANCE": config.get("rater_error_variance", None),
            "CONTEXT": context,
            "JAVASCRIPT_PATH": javascript_path,
            "OUTPUT_DIR": csvdir,
            "FIGURE_DIR": figdir,
        }

        # get the report directory which is at the same level
        # as the output and the figure directory
        reportdir = abspath(join(csvdir, "..", "report"))
        report_name = f"{config['experiment_id']}_report"
        merged_notebook_file = join(reportdir, f"{report_name}.ipynb")
        environ_config_file = join(reportdir, ".environ.json")

        # set the report directory as an environment variable
        os.environ["RSM_REPORT_DIR"] = reportdir

        # write out hidden environment JSON file
        with open(environ_config_file, "w") as out_environ_config:
            json.dump(environ_config, out_environ_config)

        # merge all the given sections
        self.logger.info("Merging sections")
        self.merge_notebooks(config["chosen_notebook_files"], merged_notebook_file)

        # run the merged notebook and save the output as
        # an HTML file in the report directory
        self.logger.info("Exporting HTML")
        self.convert_ipynb_to_html(merged_notebook_file, join(reportdir, f"{report_name}.html"))
        log_report_to_wandb(
            self.wandb_run, f"{context}_report", join(reportdir, f"{report_name}.html")
        )

    def create_comparison_report(
        self, config, csvdir_old, figdir_old, csvdir_new, figdir_new, output_dir
    ):
        """
        Generate an HTML report for comparing two rsmtool experiments.

        Parameters
        ----------
        config : configuration_parser.Configuration
            A configuration object
        csvdir_old : str
            The old experiment CSV output directory.
        figdir_old : str
            The old figure output directory
        csvdir_new : str
            The new experiment CSV output directory.
        figdir_new : str
            The old figure output directory
        output_dir : str
            The output dir for the new report.
        """
        # whether to use scaled predictions for both new and old; default to false
        use_scaled_predictions_old = config.get("use_scaled_predictions_old", False)
        use_scaled_predictions_new = config.get("use_scaled_predictions_new", False)

        environ_config = {
            "EXPERIMENT_ID_OLD": config["experiment_id_old"],
            "EXPERIMENT_ID_NEW": config["experiment_id_new"],
            "DESCRIPTION_OLD": config["description_old"],
            "DESCRIPTION_NEW": config["description_new"],
            "GROUPS_FOR_DESCRIPTIVES": config.get("subgroups", []),
            "GROUPS_FOR_EVALUATIONS": config.get("subgroups", []),
            "SCALED_OLD": use_scaled_predictions_old,
            "SCALED_NEW": use_scaled_predictions_new,
            "USE_THUMBNAILS": config.get("use_thumbnails", False),
            "JAVASCRIPT_PATH": javascript_path,
            "OUTPUT_DIR_NEW": csvdir_new,
            "FIGURE_DIR_NEW": figdir_new,
            "OUTPUT_DIR_OLD": csvdir_old,
            "FIGURE_DIR_OLD": figdir_old,
        }

        # create the output directory
        os.makedirs(output_dir, exist_ok=True)
        report_name = f"{config['comparison_id']}_report"
        merged_notebook_file = join(output_dir, f"{report_name}.ipynb")
        environ_config_file = join(output_dir, ".environ.json")

        # set the report directory as an environment variable
        os.environ["RSM_REPORT_DIR"] = abspath(output_dir)

        # write out hidden environment JSON file
        with open(environ_config_file, "w") as out_environ_config:
            json.dump(environ_config, out_environ_config)

        # merge all the given sections
        self.logger.info("Merging sections")
        self.merge_notebooks(config["chosen_notebook_files"], merged_notebook_file)

        # run the merged notebook and save the output as
        # an HTML file in the report directory
        self.logger.info("Exporting HTML")
        self.convert_ipynb_to_html(merged_notebook_file, join(output_dir, f"{report_name}.html"))
        log_report_to_wandb(
            self.wandb_run, "comparison_report", join(output_dir, f"{report_name}.html")
        )

    def create_summary_report(self, config, all_experiments, csvdir):
        """
        Generate an HTML report for summarizing the given rsmtool experiments.

        Parameters
        ----------
        config : configuration_parser.Configuration
            A configuration object
        all_experiments : list of str
            A list of experiment configuration files to summarize.
        csvdir : str
            The experiment CSV output directory.
        """
        environ_config = {
            "SUMMARY_ID": config["summary_id"],
            "DESCRIPTION": config["description"],
            "GROUPS_FOR_DESCRIPTIVES": config.get("subgroups", []),
            "GROUPS_FOR_EVALUATIONS": config.get("subgroups", []),
            "USE_THUMBNAILS": config.get("use_thumbnails", False),
            "FILE_FORMAT": config.get("file_format", "csv"),
            "JSONS": all_experiments,
            "JAVASCRIPT_PATH": javascript_path,
            "OUTPUT_DIR": csvdir,
        }

        report_name = f"{config['summary_id']}_report"
        reportdir = abspath(join(csvdir, "..", "report"))
        merged_notebook_file = join(reportdir, f"{report_name}.ipynb")
        environ_config_file = join(reportdir, ".environ.json")

        # set the report directory as an environment variable
        os.environ["RSM_REPORT_DIR"] = reportdir

        # write out hidden environment JSON file
        with open(environ_config_file, "w") as out_environ_config:
            json.dump(environ_config, out_environ_config)

        # merge all the given sections
        self.logger.info("Merging sections")
        self.merge_notebooks(config["chosen_notebook_files"], merged_notebook_file)

        # run the merged notebook and save the output as
        # an HTML file in the report directory
        self.logger.info("Exporting HTML")
        self.convert_ipynb_to_html(merged_notebook_file, join(reportdir, f"{report_name}.html"))
        log_report_to_wandb(
            self.wandb_run, "summary_report", join(reportdir, f"{report_name}.html")
        )

    def create_explanation_report(self, config, csv_dir, output_dir):
        """
        Generate a html report for rsmexplain.

        Parameters
        ----------
        config : configuration_parser.Configuration
            A configuration object
        csv_dir : str
            The experiment output directory containing CSV files with SHAP values
        output_dir : str
            The directory for the html report

        """
        # we define a directory for the saved figures
        fig_dir = normpath(join(output_dir, "..", "figure"))

        environ_config = {
            "EXPERIMENT_ID": config["experiment_id"],
            "JAVASCRIPT_PATH": javascript_path,
            "DESCRIPTION": config["description"],
            "EXPLANATION": config["explanation"],  # the path to the explanation object
            "BACKGROUND_KMEANS_SIZE": config["background_kmeans_size"],
            "IDs": config["ids"],
            "CSV_DIR": csv_dir,  # the report loads some csv files, so we need this parameter
            "NUM_FEATURES_TO_DISPLAY": config["num_features_to_display"],
            "FIG_DIR": fig_dir,
            "HAS_SINGLE_EXAMPLE": config["has_single_example"],
        }
        report_name = f"{config['experiment_id']}_explain_report"
        reportdir = abspath(join(output_dir, "..", "report"))

        merged_notebook_file = join(reportdir, f"{report_name}.ipynb")
        environ_config_file = join(reportdir, ".environ.json")

        # set the report directory as an environment variable
        os.environ["RSM_REPORT_DIR"] = reportdir

        # write out hidden environment JSON file
        with open(environ_config_file, "w") as out_environ_config:
            json.dump(environ_config, out_environ_config)

        # merge all the given sections
        self.logger.info("Merging sections")
        self.merge_notebooks(config["chosen_notebook_files"], merged_notebook_file)

        # run the merged notebook and save the output as
        # an HTML file in the report directory
        self.logger.info("Exporting HTML")
        self.convert_ipynb_to_html(merged_notebook_file, join(reportdir, f"{report_name}.html"))
        self.logger.info("Success")
        log_report_to_wandb(
            self.wandb_run, "explain_report", join(reportdir, f"{report_name}.html")
        )


def main():  # noqa: D103
    # set up an argument parser
    parser = argparse.ArgumentParser(prog="render_notebook")
    parser.add_argument("ipynb_file", help="IPython notebook file")
    parser.add_argument("html_file", help="output HTML file")

    # parse given command line arguments
    args = parser.parse_args()

    # convert notebook to HTML
    Reporter.convert_ipynb_to_html(args.ipynb_file, args.html_file)


if __name__ == "__main__":
    main()
