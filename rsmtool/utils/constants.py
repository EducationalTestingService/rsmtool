"""
Various RSMTool constants used across the codebase.

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

import re

from .models import BUILTIN_MODELS, VALID_SKLL_MODELS

DEFAULTS = {
    "id_column": "spkitemid",
    "description": "",
    "description_old": "",
    "description_new": "",
    "train_label_column": "sc1",
    "test_label_column": "sc1",
    "human_score_column": "sc1",
    "exclude_zero_scores": True,
    "use_scaled_predictions": False,
    "use_scaled_predictions_old": False,
    "use_scaled_predictions_new": False,
    "select_transformations": False,
    "standardize_features": True,
    "truncate_outliers": True,
    "use_thumbnails": False,
    "use_truncation_thresholds": False,
    "scale_with": None,
    "predict_expected_scores": False,
    "sign": None,
    "features": None,
    "length_column": None,
    "second_human_score_column": None,
    "file_format": "csv",
    "form_level_scores": None,
    "candidate_column": None,
    "general_sections": ["all"],
    "special_sections": None,
    "custom_sections": None,
    "feature_subset_file": None,
    "feature_subset": None,
    "rater_error_variance": None,
    "trim_min": None,
    "trim_max": None,
    "trim_tolerance": 0.4998,
    "subgroups": [],
    "min_n_per_group": None,
    "skll_fixed_parameters": {},
    "skll_objective": None,
    "section_order": None,
    "flag_column": None,
    "flag_column_test": None,
    "min_items_per_candidate": None,
    "experiment_names": None,
    "folds_file": None,
    "folds": 5,
    "sample_range": None,  # range of specific sample IDs to be explained
    "sample_size": None,  # size of random sample to be explained
    "sample_ids": None,  # specific sample IDs to be explained
    "background_kmeans_size": 500,  # size of k-means sample for background
    "num_features_to_display": 15,  # how many features should be displayed in rsmexplain plots
    "show_auto_cohorts": False,  # enables auto cohort plots for rsmexplain
    "skll_grid_search_jobs": 1,
    "use_wandb": False,  # enables logging to Weights & Biases
    "wandb_project": None,
    "wandb_entity": None,
}

LIST_FIELDS = [
    "general_sections",
    "special_sections",
    "custom_sections",
    "subgroups",
    "section_order",
    "experiment_dirs",
    "experiment_names",
]

BOOLEAN_FIELDS = [
    "exclude_zero_scores",
    "predict_expected_scores",
    "use_scaled_predictions",
    "use_scaled_predictions_old",
    "use_scaled_predictions_new",
    "use_thumbnails",
    "use_truncation_thresholds",
    "select_transformations",
]


CHECK_FIELDS = {
    "rsmtool": {
        "required": ["experiment_id", "model", "train_file", "test_file"],
        "optional": [
            "description",
            "features",
            "feature_subset_file",
            "feature_subset",
            "file_format",
            "sign",
            "id_column",
            "use_thumbnails",
            "train_label_column",
            "test_label_column",
            "length_column",
            "second_human_score_column",
            "flag_column",
            "flag_column_test",
            "exclude_zero_scores",
            "rater_error_variance",
            "trim_min",
            "trim_max",
            "trim_tolerance",
            "predict_expected_scores",
            "select_transformations",
            "use_scaled_predictions",
            "use_truncation_thresholds",
            "subgroups",
            "min_n_per_group",
            "general_sections",
            "custom_sections",
            "special_sections",
            "skll_fixed_parameters",
            "skll_objective",
            "section_order",
            "candidate_column",
            "standardize_features",
            "truncate_outliers",
            "min_items_per_candidate",
            "skll_grid_search_jobs",
            "use_wandb",
            "wandb_project",
            "wandb_entity",
        ],
    },
    "rsmxval": {
        "required": ["experiment_id", "model", "train_file"],
        "optional": [
            "description",
            "folds",
            "folds_file",
            "features",
            "feature_subset_file",
            "feature_subset",
            "file_format",
            "sign",
            "id_column",
            "use_thumbnails",
            "train_label_column",
            "length_column",
            "second_human_score_column",
            "flag_column",
            "flag_column_test",
            "exclude_zero_scores",
            "rater_error_variance",
            "trim_min",
            "trim_max",
            "trim_tolerance",
            "predict_expected_scores",
            "select_transformations",
            "use_scaled_predictions",
            "use_truncation_thresholds",
            "subgroups",
            "min_n_per_group",
            "skll_fixed_parameters",
            "skll_objective",
            "candidate_column",
            "standardize_features",
            "truncate_outliers",
            "min_items_per_candidate",
            "skll_grid_search_jobs",
            "use_wandb",
            "wandb_project",
            "wandb_entity",
        ],
    },
    "rsmeval": {
        "required": [
            "experiment_id",
            "predictions_file",
            "system_score_column",
            "trim_min",
            "trim_max",
        ],
        "optional": [
            "description",
            "id_column",
            "human_score_column",
            "second_human_score_column",
            "file_format",
            "flag_column",
            "exclude_zero_scores",
            "use_thumbnails",
            "scale_with",
            "trim_tolerance",
            "rater_error_variance",
            "subgroups",
            "min_n_per_group",
            "general_sections",
            "custom_sections",
            "special_sections",
            "section_order",
            "candidate_column",
            "min_items_per_candidate",
            "use_wandb",
            "wandb_project",
            "wandb_entity",
        ],
    },
    "rsmpredict": {
        "required": ["experiment_id", "experiment_dir", "input_features_file"],
        "optional": [
            "id_column",
            "candidate_column",
            "file_format",
            "predict_expected_scores",
            "human_score_column",
            "second_human_score_column",
            "standardize_features",
            "truncate_outliers",
            "subgroups",
            "flag_column",
            "use_wandb",
            "wandb_project",
            "wandb_entity",
        ],
    },
    "rsmcompare": {
        "required": [
            "comparison_id",
            "experiment_id_old",
            "experiment_dir_old",
            "experiment_id_new",
            "experiment_dir_new",
            "description_old",
            "description_new",
        ],
        "optional": [
            "use_scaled_predictions_old",
            "use_scaled_predictions_new",
            "subgroups",
            "use_thumbnails",
            "general_sections",
            "custom_sections",
            "special_sections",
            "section_order",
            "use_wandb",
            "wandb_project",
            "wandb_entity",
        ],
    },
    "rsmsummarize": {
        "required": ["summary_id", "experiment_dirs"],
        "optional": [
            "description",
            "experiment_names",
            "file_format",
            "general_sections",
            "custom_sections",
            "use_thumbnails",
            "special_sections",
            "subgroups",
            "section_order",
            "use_wandb",
            "wandb_project",
            "wandb_entity",
        ],
    },
    "rsmexplain": {
        "required": [
            "background_data",
            "explain_data",
            "experiment_id",
            "experiment_dir",
        ],
        "optional": [
            "description",
            "id_column",
            "background_kmeans_size",
            "num_features_to_display",
            "sample_range",
            "sample_size",
            "sample_ids",
            "show_auto_cohorts",
            "standardize_features",
            "truncate_outliers",
            "general_sections",
            "custom_sections",
            "special_sections",
            "use_wandb",
            "wandb_project",
            "wandb_entity",
        ],
    },
}

POSSIBLE_EXTENSIONS = ["csv", "xlsx", "tsv"]

ID_FIELDS = {
    "rsmtool": "experiment_id",
    "rsmeval": "experiment_id",
    "rsmcompare": "comparison_id",
    "rsmsummarize": "summary_id",
    "rsmpredict": "experiment_id",
    "rsmxval": "experiment_id",
    "rsmexplain": "experiment_id",
}

CONFIGURATION_DOCUMENTATION_SLUGS = {
    "rsmtool": "usage_rsmtool.html#experiment-configuration-file",
    "rsmeval": "advanced_usage.html#experiment-configuration-file",
    "rsmcompare": "advanced_usage.html#config-file-rsmcompare",
    "rsmpredict": "advanced_usage.html#config-file-rsmpredict",
    "rsmsummarize": "advanced_usage.html#config-file-rsmsummarize",
    "rsmxval": "advanced_usage.html#config-file-rsmxval",
    "rsmexplain": "advanced_usage.html#config-file-rsmexplain",
}

VALID_PARSER_SUBCOMMANDS = ["generate", "run"]

INTERACTIVE_MODE_METADATA = {
    "experiment_id": {"label": "Experiment ID", "type": "id"},
    "comparison_id": {"label": "Comparison ID", "type": "id"},
    "summary_id": {"label": "Summary ID", "type": "id"},
    "model": {
        "label": "Model to use",
        "type": "choice",
        "choices": sorted(set(BUILTIN_MODELS + VALID_SKLL_MODELS)),
    },
    "train_file": {"label": "Path to training data file", "type": "file"},
    "test_file": {"label": "Path to evaluation data file", "type": "file"},
    "folds": {
        "label": "Number of cross-validation folds to use (<u>5</u>)",
        "type": "integer",
    },
    "folds_file": {
        "label": "Optional file with custom folds (overrides # of folds)",
        "type": "file",
    },
    "predictions_file": {
        "label": "Path to file containing predictions",
        "type": "file",
    },
    "system_score_column": {"label": "Name of column containing predictions"},
    "trim_min": {"label": "The lowest possible human score", "type": "integer"},
    "trim_max": {"label": "The highest possible human score", "type": "integer"},
    "experiment_dir": {
        "label": "Path to the directory containing RSMTool experiment",
        "type": "dir",
    },
    "input_features_file": {
        "label": "Path to input file containing features",
        "type": "file",
    },
    "experiment_id_old": {"label": "ID for old RSMTool experiment"},
    "experiment_dir_old": {"label": "Path to old RSMTool experiment", "type": "dir"},
    "description_old": {"label": "Description of old RSMTool experiment"},
    "experiment_id_new": {"label": "ID for new RSMTool experiment"},
    "experiment_dir_new": {"label": "Path to new RSMTool experiment", "type": "dir"},
    "description_new": {"label": "Description of new RSMTool experiment"},
    "experiment_dirs": {
        "label": "Paths to directories containing RSMTool experiments",
        "type": "dir",
        "count": "multiple",
    },
    "description": {"label": "Description of experiment"},
    "file_format": {
        "label": "Format for intermediate files (<u>csv</u>/tsv/xlsx)",
        "type": "format",
    },
    "id_column": {"label": "Name of column that contains response IDs (<u>spkitemid</u>)"},
    "use_thumbnails": {
        "label": "Use clickable thumbnails in report instead "
        "of full-sized images? (true/<u>false</u>)",
        "type": "boolean",
    },
    "train_label_column": {
        "label": "Name of column in training data that " "contains human scores (<u>sc1</u>)"
    },
    "test_label_column": {
        "label": "Name of column in evaluation data that " "contains human scores (<u>sc1</u>)"
    },
    "length_column": {
        "label": "Name of column in training/evaluation data "
        "that contains response lengths, if any"
    },
    "human_score_column": {
        "label": "Name of column in evaluation data " "that contains human scores (<u>sc1</u>)"
    },
    "second_human_score_column": {
        "label": "Name of column in evaluation "
        "data that contains scores from "
        "a second human, if any"
    },
    "exclude_zero_scores": {
        "label": "Keep responses with human scores of 0 in "
        "training/evaluation data "
        "(true/<u>false</u>)",
        "type": "boolean",
    },
    "use_scaled_predictions": {
        "label": "Use scaled predictions instead of "
        "raw in report analyses "
        "(true/<u>false</u>)",
        "type": "boolean",
    },
    "standardize_features": {
        "label": "Standardize all features " "(<u>true</u>/false)",
        "type": "boolean",
    },
    "subgroups": {
        "label": "List of column names containing subgroup variables",
        "count": "multiple",
    },
    "background_data": {
        "label": "Path to file to be used as background distribution ",
        "type": "file",
    },
    "background_kmeans_size": {
        "label": "size of k-means sample for background (<u>500</u>)",
        "type": "integer",
    },
    "explain_data": {"label": "Path to file to be explained ", "type": "file"},
    "sample_range": {"label": "Range of specific row IDs to explain "},
    "sample_size": {
        "label": "Size of random sample to be explained ",
        "type": "integer",
    },
    "num_features_to_display": {
        "label": "Number of features to be displayed in plots (<u>15</u>)",
        "type": "integer",
    },
    "show_auto_cohorts": {
        "label": "Show auto cohorts plot (true/<u>false</u>)",
        "type": "boolean",
    },
}

# regular expression used to parse rsmexplain range values
RSMEXPLAIN_RANGE_REGEXP = re.compile(r"^(?P<start>[0-9]+)\-(?P<end>[0-9]+)$")
