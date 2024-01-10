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
            "section_order",
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

# replacement dictionary for intermediate file names to descriptive names
INTERMEDIATE_FILES_TO_DESCRIPTIONS = {
    "rsmtool": {
        "betas": "Standardized & relative regression coefficients",
        "coefficients": "Feature coefficients (including intercept)",
        "coefficients_scaled": "Scaled feature coefficients (including intercept)",
        "confMatrix": "Confusion matrix between human scores and machine scores",
        "confMatrix_h1h2": "Confusion matrix between human-human scores",
        "consistency": "Human-human agreement metrics",
        "consistency_by_ZZZ": "Human-human agreement by ZZZ subgroup",
        "cors_orig": "Correlations between raw features & between features and human score",
        "cors_processed": "Correlations between pre-processed features & between features and human score",
        "data_composition": "Data composition statistics",
        "data_composition_by_ZZZ": "Data composition statistics by ZZZ subgroup",
        "degradation": "Degradation in H-M performance compared to H-H",
        "disattenuated_correlations": "Disattenuated correlations between various scores",
        "disattenuated_correlations_by_ZZZ": "Disattenuated correlations by ZZZ subgroup",
        "estimates_csd_by_ZZZ": "Conditional score differences by ZZZ subgroup",
        "estimates_osa_by_ZZZ": "Overall score accuracy by ZZZ subgroup",
        "estimates_osd_by_ZZZ": "Overall score differences by ZZZ subgroup",
        "eval": "Full set of evaluation metrics",
        "eval_by_ZZZ": "Evaluation metrics by ZZZ subgroup",
        "eval_short": "Subset of more commonly used evaluation metrics",
        "fairness_metrics_by_ZZZ": "R^2 and p-values for all fairness models by ZZZ subgroup",
        "feature": "Feature values, signs, and transforms used in the final model",
        "feature_descriptives": "Main Descriptive statistics for raw feature values",
        "feature_descriptivesExtra": "Additional descriptive statistics for raw feature values",
        "feature_outliers": "Number and percentage of outliers trunctated during feature pre-processing",
        "margcor_length_all_data": "Marginal correlations between each pre-processed feature and length",
        "margcor_length_by_ZZZ": "Marginal feature-length correlations by ZZZ subgroup",
        "margcor_score_all_data": "Marginal correlations between each pre-processed feature and human score",
        "margcor_score_by_ZZZ": "Marginal feature-score correlations by ZZZ subgroup",
        "model_fit": "R^2 and adjusted R^2 computed on the training set",
        "pca": "Results of principal components analysis on the pre-processed feature values",
        "pcavar": "Eigenvalues and variance explained by each principal component",
        "pcor_length_all_data": "Partial correlations between each pre-processed feature and length, controlling for other features",
        "pcor_length_by_ZZZ": "Partial feature-length correlations by ZZZ subgroup, controlling for other features",
        "pcor_score_all_data": "Partial correlations between each pre-processed feature & human score, controlling for other features",
        "pcor_score_no_length_all_data": "Partial feature-score correlations, also controlling for length",
        "pcor_score_by_ZZZ": "Partial feature-score correlations by ZZZ subgroup, controlling for other features",
        "pcor_score_no_length_by_ZZZ": "Partial feature-score correlations by ZZZ subgroup, also controlling for length",
        "postprocessing_params": "Parameters used for trimming and scaling predicted scores",
        "pred_processed": "Predicted scores for the evaluation set",
        "pred_train": "Predicted scores for the training set",
        "score_dist": "Distributions of human and machine scores",
        "test_excluded_composition": "Composition of evaluation set responses excluded from analysis",
        "test_excluded_responses": "Actual evaluation set responses excluded from analysis",
        "test_features": "Raw feature values for the evaluation set",
        "test_human_scores": "Human scores for the evaluation set",
        "test_metadata": "Metadata columns for the evaluation set",
        "test_other_columns": "Unused columns for the evaluation set",
        "test_preprocessed_features": "Pre-processed feature values for the evaluation set",
        "test_responses_with_excluded_flags": "Evaluation set responses filtered out via `flag_column`",
        "train_excluded_composition": "Composition of training set responses excluded from analysis",
        "train_excluded_responses": "Actual training set responses excluded from analysis",
        "train_features": "Raw feature values for the training set",
        "train_metadata": "Metadata columns for the training set",
        "train_missing_feature_values": "Number of non-numeric values for each feature in training set",
        "train_other_columns": "Unused columns for the training set",
        "train_preprocessed_features": "Pre-processed feature values for the training set",
        "train_response_lengths": "Values of `length` column for training set",
        "train_responses_with_excluded_flags": "Training set responses filtered out via `flag_column`",
        "true_score_eval": "Metrics evaluating how well system scores predict true scores",
    },
    "rsmsummarize": {
        "betas": "Standardized coefficients for each experiment",
        "eval_short": "More commonly used evaluation metrics for each experiment",
        "margcor_score_all_data": "Marginal correlations between pre-processed features and human score for each experiment",
        "model_fit": "R^2 and adjusted R^2 computed on the training set for each experiment",
        "model_summary": "Summary of models used for each experiment",
        "pcor_score_all_data": "Partial feature-score correlations features for each experiment, controlling for other features",
        "pcor_score_no_length_all_data": "Partial feature-score correlations features for each experiment, also controlling for length",
        "true_score_eval": "Metrics evaluating how well system scores predict true scores for each experiment",
    },
}
