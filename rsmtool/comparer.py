"""
Classes for comparing outputs of two RSMTool experiments.

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

import warnings
from collections import defaultdict
from copy import deepcopy
from os.path import exists, join
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from .reader import DataReader
from .utils.files import get_output_directory_extension

_df_eval_columns_existing_raw = [
    "N",
    "h_mean",
    "h_sd",
    "sys_mean.raw_trim",
    "sys_sd.raw_trim",
    "corr.raw_trim",
    "SMD.raw_trim",
    "sys_mean.raw_trim_round",
    "sys_sd.raw_trim_round",
    "exact_agr.raw_trim_round",
    "kappa.raw_trim_round",
    "wtkappa.raw_trim",
    "adj_agr.raw_trim_round",
    "SMD.raw_trim_round",
    "R2.raw_trim",
    "RMSE.raw_trim",
]

_df_eval_columns_existing_scale = [
    "N",
    "h_mean",
    "h_sd",
    "sys_mean.scale_trim",
    "sys_sd.scale_trim",
    "corr.scale_trim",
    "SMD.scale_trim",
    "sys_mean.scale_trim_round",
    "sys_sd.scale_trim_round",
    "exact_agr.scale_trim_round",
    "kappa.scale_trim_round",
    "wtkappa.scale_trim",
    "adj_agr.scale_trim_round",
    "SMD.scale_trim_round",
    "R2.scale_trim",
    "RMSE.scale_trim",
]


_df_eval_columns_renamed = [
    "N",
    "H1 mean",
    "H1 SD",
    "score mean(b)",
    "score SD(b)",
    "Pearson(b)",
    "SMD(b)",
    "score mean(br)",
    "score SD(br)",
    "Agmt.(br)",
    "K(br)",
    "QWK(b)",
    "Adj. Agmt.(br)",
    "SMD(br)",
    "R2(b)",
    "RMSE(b)",
]

raw_rename_dict = dict(zip(_df_eval_columns_existing_raw, _df_eval_columns_renamed))
scale_rename_dict = dict(zip(_df_eval_columns_existing_scale, _df_eval_columns_renamed))


class Comparer:
    """Class to perform comparisons between two RSMTool experiments."""

    @staticmethod
    def _modify_eval_columns_to_ensure_version_compatibilty(
        df: pd.DataFrame,
        rename_dict: Dict[str, str],
        existing_eval_cols: List[str],
        short_metrics_list: List[str],
    ) -> Tuple[Dict[str, str], List[str], List[str], str]:
        """
        Ensure that column names in eval data frames are backwards compatible.

        This helper method ensures that the column names for eval data frames
        are forward and backward compatible. There are two major changes in
        RSMTool (7.0 or greater) that necessitate this: (1) for subgroup
        calculations, 'DSM' is now used instead of 'SMD', and (2) QWK
        statistics are now calculated on un-rounded scores. Thus, we need to
        check these columns to see which sets of names are being used.

        Parameters
        ----------
        df : pandas.DataFrame
            The evaluation data frame.
        rename_dict : Dict[str, str]
            The rename dictionary.
        existing_eval_cols : List[str]
            The existing evaluation columns.
        short_metrics_list : List[str]
            The list of columns for the short metrics file.

        Returns
        -------
        rename_dict_new : Dict[str, str]
            The updated rename dictionary.
        existing_eval_cols_new : List[str]
            The updated existing evaluation columns
        short_metrics_list_new : List[str]
            The updated list of columns for the short metrics file.
        smd_name : str
            The SMD column name (either 'SMD' or 'DSM')
        """
        rename_dict_new = deepcopy(rename_dict)
        existing_eval_cols_new = deepcopy(existing_eval_cols)
        short_metrics_list_new = deepcopy(short_metrics_list)
        smd_name = "SMD"

        # previously, the QWK metric used `trim_round` scores; now, we use just `trim` scores;
        # We check whether `trim_round` scores were used and
        # raise an error if this is the case
        if any(col.endswith("trim_round") for col in df if col.startswith("wtkappa")):
            raise ValueError(
                "RSMTool (7.0 or greater) uses "
                "unrounded scores for weighted kappa calculations. At least one "
                "of your experiments was run "
                "using an older version of RSMTool that used  "
                "rounded scores instead. Please re-run "
                "the experiment with the latest version of RSMTool."
            )

        # we check if DSM was calculated (which is what we expect for subgroup evals),
        # and if so, we update the rename dictionary and column lists accordingly; we
        # also set `smd_name` equal to 'DSM'
        if any(col.startswith("DSM") for col in df):
            smd_name = "DSM"
            existing_eval_cols_new = [col.replace("SMD", "DSM") for col in existing_eval_cols_new]
            rename_dict_new = {
                key.replace("SMD", "DSM"): val.replace("SMD", "DSM")
                for key, val in rename_dict_new.items()
            }

        return (
            rename_dict_new,
            existing_eval_cols_new,
            short_metrics_list_new,
            smd_name,
        )

    @staticmethod
    def make_summary_stat_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute summary statistics for the data in the given frame.

        Parameters
        ----------
        df : pandas.DataFrame
            Data frame containing numeric data.

        Returns
        -------
        res : pandas.DataFrame
            Data frame containing summary statistics for data
            in the input frame.
        """
        series = []
        for summary_func in [np.mean, np.std, np.median, np.min, np.max]:
            # apply function, but catch and ignore warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                series.append(df.apply(summary_func))

        res = pd.concat(series, axis=1, sort=True)
        res.columns = ["MEAN", "SD", "MEDIAN", "MIN", "MAX"]
        return res

    @staticmethod
    def compute_correlations_between_versions(
        df_old: pd.DataFrame,
        df_new: pd.DataFrame,
        human_score: str = "sc1",
        id_column: str = "spkitemid",
    ) -> pd.DataFrame:
        """
        Compute correlations between old and new feature values.

        This method computes correlations between old and new feature values
        in the two given frames as well as the correlations between each
        feature value and the human score.

        Parameters
        ----------
        df_old : pandas.DataFrame
            Data frame with feature values for the 'old' model.
        df_new : pandas.DataFrame
            Data frame with feature values for the 'new' model.
        human_score : str
            Name of the column containing human score. Must be the same for both
            data sets.
            Defaults to ``"sc1"``.
        id_column : str
            Name of the column containing id for each response. Must be the same
            for both data sets.
            Defaults to ``"spkitemid"``.

        Returns
        -------
        df_correlations: pandas.DataFrame
            Data frame with a row for each feature and the following columns:
                - "N": total number of responses
                - "human_old": correlation with human score in the old frame
                - "human_new": correlation with human score in the new frame
                - "old_new": correlation between old and new frames

        Raises
        ------
        ValueError
            If there are no shared features between the two sets.
        ValueError
            If there are no shared responses between the two sets.
        """
        # Only use features that appear in both datasets
        features_old = [column for column in df_old if column not in [id_column, human_score]]
        features_new = [column for column in df_new if column not in [id_column, human_score]]

        features = list(set(features_old).intersection(features_new))

        if len(features) == 0:
            raise ValueError("There are no matching features " "in these two data sets.")

        columns = features + [id_column, human_score]

        # merge the two data sets and display a warning
        # if there are non-matching ids
        df_merged = pd.merge(
            df_old[columns],
            df_new[columns],
            on=[id_column],
            suffixes=["%%%old", "%%%new"],
        )

        if len(df_merged) == 0:
            raise ValueError("There are no shared ids between these two datasets.")

        if len(df_merged) != len(df_old):
            warnings.warn(
                "Some responses from the old data "
                "were not present in the new data and therefore "
                "were excluded from the analysis."
            )

        if len(df_merged) != len(df_new):
            warnings.warn(
                "Some responses from the new data "
                "were not present in the old data and therefore "
                "were excluded from the analysis."
            )

        # compute correlations between each feature and human score.
        # we are using the same approach as used in analysis.py
        correlation_list = []
        for feature in features:
            # compute correlations
            df_cor = pd.DataFrame(
                {
                    "Feature": [feature],
                    "N": len(df_merged),
                    "human_old": pearsonr(
                        df_merged[f"{human_score}%%%old"], df_merged[f"{feature}%%%old"]
                    )[0],
                    "human_new": pearsonr(
                        df_merged[f"{human_score}%%%new"], df_merged[f"{feature}%%%new"]
                    )[0],
                    "old_new": pearsonr(
                        df_merged[f"{feature}%%%new"], df_merged[f"{feature}%%%old"]
                    )[0],
                }
            )
            correlation_list.append(df_cor)

        df_correlations = pd.concat(correlation_list, sort=True)
        df_correlations.index = df_correlations["Feature"]
        df_correlations.index.name = None

        return df_correlations

    @staticmethod
    def process_confusion_matrix(conf_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Add "human" and "machine" to column names in the confusion matrix.

        Parameters
        ----------
        conf_matrix : pandas.DataFrame
            data frame containing the confusion matrix.

        Returns
        -------
        conf_matrix_renamed : pandas.DataFrame
            pandas Data Frame containing the confusion matrix with the columns
            renamed.
        """
        conf_matrix_renamed = conf_matrix.copy()
        conf_matrix_renamed.index = [f"machine {n}" for n in conf_matrix.index]
        conf_matrix_renamed.columns = [f"human {x}" for x in conf_matrix.columns]
        return conf_matrix_renamed

    def load_rsmtool_output(
        self, filedir: str, figdir: str, experiment_id: str, prefix: str, groups_eval: List[str]
    ) -> Tuple[Dict[str, pd.DataFrame], Dict[str, str], str]:
        """
        Load all of the outputs of an rsmtool experiment.

        For each type of output, we first check whether the file exists
        to allow comparing experiments with different sets of outputs.

        Parameters
        ----------
        filedir : str
            Path to the directory containing output files.
        figdir : str
            Path to the directory containing output figures.
        experiment_id : str
            Original ``experiment_id`` used to generate the output files.
        prefix : str
            Must be set to ``"scale"`` or ``"raw"``. Indicates whether the score
            is scaled or not.
        groups_eval: List[str]
            List of subgroup names used for subgroup evaluation.

        Returns
        -------
        files : Dict[str, pd.DataFrame]
            A dictionary mapping data frame names converted to the actual pandas
            data frames. If a particular type of output did not exist for the
            experiment, its value will be an empty data frame.
        figs: Dict[str, str]
            A dictionary mapping figure names to the paths of the files
            containing the figures.
        file_format: str
            The file format used for the output files, e.g., csv, tsv, or xlsx.
        """
        file_format = get_output_directory_extension(filedir, experiment_id)

        files: Dict[str, pd.DataFrame] = defaultdict(pd.DataFrame)
        figs = {}

        # feature distributions and the inter-feature correlations
        feature_train_file = join(filedir, f"{experiment_id}_train_features.{file_format}")
        if exists(feature_train_file):
            files["df_train_features"] = DataReader.read_from_file(feature_train_file)

        feature_distplots_file = join(figdir, f"{experiment_id}_distrib.svg")
        if exists(feature_distplots_file):
            figs["feature_distplots"] = feature_distplots_file

        feature_cors_file = join(filedir, f"{experiment_id}_cors_processed.{file_format}")
        if exists(feature_cors_file):
            files["df_feature_cors"] = DataReader.read_from_file(feature_cors_file, index_col=0)

        # df_scores
        scores_file = join(filedir, f"{experiment_id}_pred_processed.{file_format}")
        if exists(scores_file):
            df_scores = DataReader.read_from_file(scores_file, converters={"spkitemid": str})
            files["df_scores"] = df_scores[["spkitemid", "sc1", prefix]]

        # model coefficients if present
        betas_file = join(filedir, f"{experiment_id}_betas.{file_format}")
        if exists(betas_file):
            files["df_coef"] = DataReader.read_from_file(betas_file, index_col=0)
            files["df_coef"].index.name = None

        # read in the model fit files if present
        model_fit_file = join(filedir, f"{experiment_id}_model_fit.{file_format}")
        if exists(model_fit_file):
            files["df_model_fit"] = DataReader.read_from_file(model_fit_file)

        # human human agreement
        consistency_file = join(filedir, f"{experiment_id}_consistency.{file_format}")

        # load if consistency file is present
        if exists(consistency_file):
            df_consistency = DataReader.read_from_file(consistency_file, index_col=0)
            files["df_consistency"] = df_consistency

        # degradation
        degradation_file = join(filedir, f"{experiment_id}_degradation.{file_format}")

        # load if degradation file is present
        if exists(degradation_file):
            df_degradation = DataReader.read_from_file(degradation_file, index_col=0)
            files["df_degradation"] = df_degradation

        # disattenuated correlations
        dis_corr_file = join(filedir, f"{experiment_id}_disattenuated_correlations.{file_format}")

        # load if disattenuated correlations is present
        if exists(dis_corr_file):
            df_dis_corr = DataReader.read_from_file(dis_corr_file, index_col=0)
            # we only use the row for raw_trim or scale_trim score
            files["df_disattenuated_correlations"] = df_dis_corr.loc[[f"{prefix}_trim"]]

        # read in disattenuated correlations by group
        for group in groups_eval:
            group_dis_corr_file = join(
                filedir,
                f"{experiment_id}_disattenuated_correlations_by_{group}.{file_format}",
            )
            if exists(group_dis_corr_file):
                df_dis_cor_group = DataReader.read_from_file(group_dis_corr_file, index_col=0)
                files[f"df_disattenuated_correlations_by_{group}"] = df_dis_cor_group
                files[f"df_disattenuated_correlations_by_{group}_overview"] = (
                    self.make_summary_stat_df(df_dis_cor_group)
                )

        # true score evaluations
        true_score_eval_file = join(filedir, f"{experiment_id}_true_score_eval.{file_format}")

        # load true score evaluations if present
        if exists(true_score_eval_file):
            df_true_score_eval = DataReader.read_from_file(true_score_eval_file, index_col=0)
            # we only use the row for raw_trim or scale_trim score
            files["df_true_score_eval"] = df_true_score_eval.loc[[f"{prefix}_trim"]]

        # use the raw columns or the scale columns depending on the prefix
        existing_eval_cols = (
            _df_eval_columns_existing_raw if prefix == "raw" else _df_eval_columns_existing_scale
        )
        rename_dict = raw_rename_dict if prefix == "raw" else scale_rename_dict

        # read in the short version of the evaluation metrics for all data
        short_metrics_list = [
            "N",
            "Adj. Agmt.(br)",
            "Agmt.(br)",
            "K(br)",
            "Pearson(b)",
            "QWK(b)",
            "R2(b)",
            "RMSE(b)",
        ]
        eval_file_short = join(filedir, f"{experiment_id}_eval_short.{file_format}")

        if exists(eval_file_short):
            df_eval = DataReader.read_from_file(eval_file_short, index_col=0)

            (
                rename_dict_new,
                existing_eval_cols_new,
                short_metrics_list_new,
                _,
            ) = self._modify_eval_columns_to_ensure_version_compatibilty(
                df_eval, rename_dict, existing_eval_cols, short_metrics_list
            )

            df_eval = df_eval[existing_eval_cols_new]
            df_eval = df_eval.rename(columns=rename_dict_new)
            files["df_eval"] = df_eval[short_metrics_list_new]
            files["df_eval"].index.name = None

        eval_file = join(filedir, f"{experiment_id}_eval.{file_format}")
        if exists(eval_file):
            files["df_eval_for_degradation"] = DataReader.read_from_file(eval_file, index_col=0)

        # read in the evaluation metrics by subgroup, if we are asked to
        for group in groups_eval:
            group_eval_file = join(filedir, f"{experiment_id}_eval_by_{group}.{file_format}")
            if exists(group_eval_file):
                df_eval = DataReader.read_from_file(group_eval_file, index_col=0)

                (
                    rename_dict_new,
                    existing_eval_cols_new,
                    short_metrics_list_new,
                    smd_name,
                ) = self._modify_eval_columns_to_ensure_version_compatibilty(
                    df_eval,
                    rename_dict,
                    existing_eval_cols,
                    short_metrics_list,
                )

                df_eval = df_eval[existing_eval_cols_new]
                df_eval = df_eval.rename(columns=rename_dict_new)
                files[f"df_eval_by_{group}"] = df_eval[short_metrics_list_new]
                files[f"df_eval_by_{group}"].index.name = None

                series = files[f"df_eval_by_{group}"]
                files[f"df_eval_by_{group}_overview"] = self.make_summary_stat_df(series)

                # set the ordering of mean/SD/SMD statistics
                files[f"df_eval_by_{group}_m_sd"] = df_eval[
                    [
                        "N",
                        "H1 mean",
                        "H1 SD",
                        "score mean(br)",
                        "score SD(br)",
                        "score mean(b)",
                        "score SD(b)",
                        f"{smd_name}(br)",
                        f"{smd_name}(b)",
                    ]
                ]
                files[f"df_eval_by_{group}_m_sd"].index.name = None

        # read in the partial correlations vs. score for all data
        pcor_score_file = join(filedir, f"{experiment_id}_pcor_score_all_data.{file_format}")
        if exists(pcor_score_file):
            files["df_pcor_sc1"] = DataReader.read_from_file(pcor_score_file, index_col=0)
            files["df_pcor_sc1_overview"] = self.make_summary_stat_df(files["df_pcor_sc1"])

        # read in the partial correlations by subgroups, if we are asked to
        for group in groups_eval:
            group_pcor_file = join(filedir, f"{experiment_id}_pcor_score_by_{group}.{file_format}")
            if exists(group_pcor_file):
                files[f"df_pcor_sc1_by_{group}"] = DataReader.read_from_file(
                    group_pcor_file, index_col=0
                )

                series = files[f"df_pcor_sc1_by_{group}"]
                files[f"df_pcor_sc1_{group}_overview"] = self.make_summary_stat_df(series)

        # read in the marginal correlations vs. score for all data
        mcor_score_file = join(filedir, f"{experiment_id}_margcor_score_all_data.{file_format}")
        if exists(mcor_score_file):
            files["df_mcor_sc1"] = DataReader.read_from_file(mcor_score_file, index_col=0)
            files["df_mcor_sc1_overview"] = self.make_summary_stat_df(files["df_mcor_sc1"])

        # read in the partial correlations by subgroups, if we are asked to
        for group in groups_eval:
            group_mcor_file = join(
                filedir, f"{experiment_id}_margcor_score_by_{group}.{file_format}"
            )
            if exists(group_mcor_file):
                files[f"df_mcor_sc1_by_{group}"] = DataReader.read_from_file(
                    group_mcor_file, index_col=0
                )

                series = files[f"df_mcor_sc1_by_{group}"]
                files[f"df_mcor_sc1_{group}_overview"] = self.make_summary_stat_df(series)

        pca_file = join(filedir, f"{experiment_id}_pca.{file_format}")
        if exists(pca_file):
            files["df_pca"] = DataReader.read_from_file(pca_file, index_col=0)
            files["df_pcavar"] = DataReader.read_from_file(
                join(filedir, f"{experiment_id}_pcavar.{file_format}"), index_col=0
            )

        descriptives_file = join(filedir, f"{experiment_id}_feature_descriptives.{file_format}")
        if exists(descriptives_file):
            # we read all files pertaining to the descriptive analysis together
            # since we merge the outputs
            files["df_descriptives"] = DataReader.read_from_file(descriptives_file, index_col=0)

            # this df contains only the number of features. this is used later
            # for another two tables to show the number of features
            df_features_n_values = files["df_descriptives"][["N", "min", "max"]]

            files["df_descriptives"] = files["df_descriptives"][
                ["N", "mean", "std. dev.", "skewness", "kurtosis"]
            ]

            outliers_file = join(filedir, f"{experiment_id}_feature_outliers.{file_format}")
            df_outliers = DataReader.read_from_file(outliers_file, index_col=0)
            df_outliers = df_outliers.rename(
                columns={
                    "upper": "Upper",
                    "lower": "Lower",
                    "both": "Both",
                    "upperperc": "Upper %",
                    "lowerperc": "Lower %",
                    "bothperc": "Both %",
                }
            )
            df_outliers_columns = df_outliers.columns.tolist()
            files["df_outliers"] = df_outliers

            # join with df_features_n_values to get the value of N
            files["df_outliers"] = pd.merge(
                files["df_outliers"],
                df_features_n_values,
                left_index=True,
                right_index=True,
            )[["N"] + df_outliers_columns]

            # join with df_features_n_values to get the value of N
            percentiles_file = join(
                filedir, f"{experiment_id}_feature_descriptivesExtra.{file_format}"
            )

            files["df_percentiles"] = DataReader.read_from_file(percentiles_file, index_col=0)
            files["df_percentiles"] = pd.merge(
                files["df_percentiles"],
                df_features_n_values,
                left_index=True,
                right_index=True,
            )

            mild_outliers = (
                files["df_percentiles"]["Mild outliers"]
                / files["df_percentiles"]["N"].astype(float)
                * 100
            )

            files["df_percentiles"]["Mild outliers (%)"] = mild_outliers

            extreme_outliers = (
                files["df_percentiles"]["Extreme outliers"]
                / files["df_percentiles"]["N"].astype(float)
                * 100
            )

            files["df_percentiles"]["Extreme outliers (%)"] = extreme_outliers

            files["df_percentiles"] = files["df_percentiles"][
                [
                    "N",
                    "min",
                    "max",
                    "1%",
                    "5%",
                    "25%",
                    "50%",
                    "75%",
                    "95%",
                    "99%",
                    "IQR",
                    "Mild outliers",
                    "Mild outliers (%)",
                    "Extreme outliers",
                    "Extreme outliers (%)",
                ]
            ]

        confmatrix_file = join(filedir, f"{experiment_id}_confMatrix.{file_format}")
        if exists(confmatrix_file):
            conf_matrix = DataReader.read_from_file(confmatrix_file, index_col=0)
            files["df_confmatrix"] = self.process_confusion_matrix(conf_matrix)

        score_dist_file = join(filedir, f"{experiment_id}_score_dist.{file_format}")
        if exists(score_dist_file):
            df_score_dist = DataReader.read_from_file(score_dist_file, index_col=1)
            df_score_dist.rename(columns={f"sys_{prefix}": "sys"}, inplace=True)
            files["df_score_dist"] = df_score_dist[["human", "sys", "difference"]]

        # read in the feature boxplots by subgroup, if we were asked to
        for group in groups_eval:
            feature_boxplot_prefix = join(figdir, f"{experiment_id}_feature_boxplot_by_{group}")
            svg_file = join(feature_boxplot_prefix + ".svg")
            png_file = join(feature_boxplot_prefix + ".png")
            if exists(svg_file):
                figs[f"feature_boxplots_by_{group}_svg"] = svg_file

            elif exists(png_file):
                figs[f"feature_boxplots_by_{group}_png"] = png_file

        # read in the betas image if exists
        betas_svg = join(figdir, f"{experiment_id}_betas.svg")
        if exists(betas_svg):
            figs["betas"] = betas_svg

        # read in the evaluation barplots by subgroup, if we were asked to
        for group in groups_eval:
            eval_barplot_svg_file = join(figdir, f"{experiment_id}_eval_by_{group}.svg")
            if exists(eval_barplot_svg_file):
                figs[f"eval_barplot_by_{group}"] = eval_barplot_svg_file

        pca_svg_file = join(figdir, f"{experiment_id}_pca.svg")
        if exists(pca_svg_file):
            figs["pca_scree_plot"] = pca_svg_file

        return (files, figs, file_format)
