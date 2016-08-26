"""
Functions for comparing outputs of two rsmtool experiments.

:author: Nitin Madnani (nmadnani@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:organization: ETS
"""

import base64
import numpy as np
import pandas as pd
import warnings

from collections import defaultdict
from scipy.stats import pearsonr
from os.path import exists, join



_df_eval_columns_existing_raw = ["N", "h_mean", "h_sd",
                                 "sys_mean.raw_trim",
                                 "sys_sd.raw_trim",
                                 "corr.raw_trim",
                                 "SMD.raw_trim",
                                 "sys_mean.raw_trim_round",
                                 "sys_sd.raw_trim_round",
                                 "exact_agr.raw_trim_round",
                                 "kappa.raw_trim_round",
                                 "wtkappa.raw_trim_round",
                                 "adj_agr.raw_trim_round",
                                 "SMD.raw_trim_round",
                                 "R2.raw_trim",
                                 "RMSE.raw_trim"]

_df_eval_columns_existing_scale = ["N", "h_mean", "h_sd",
                                   "sys_mean.scale_trim",
                                   "sys_sd.scale_trim",
                                   "corr.scale_trim",
                                   "SMD.scale_trim",
                                   "sys_mean.scale_trim_round",
                                   "sys_sd.scale_trim_round",
                                   "exact_agr.scale_trim_round",
                                   "kappa.scale_trim_round",
                                   "wtkappa.scale_trim_round",
                                   "adj_agr.scale_trim_round",
                                   "SMD.scale_trim_round",
                                   "R2.scale_trim",
                                   "RMSE.scale_trim"]


_df_eval_columns_renamed = ["N", "H1 mean", "H1 SD",
                            "score mean(b)",
                            "score SD(b)",
                            "Pearson(b)",
                            "SMD(b)",
                            "score mean(br)",
                            "score SD(br)",
                            "Agmt.(br)",
                            "K(br)",
                            "QWK(br)",
                            "Adj. Agmt.(br)",
                            "SMD(br)",
                            "R2(b)",
                            "RMSE(b)"]
raw_renamedict = dict(zip(_df_eval_columns_existing_raw, _df_eval_columns_renamed))
scale_renamedict = dict(zip(_df_eval_columns_existing_scale, _df_eval_columns_renamed))


def make_summary_stat_df(df):
    """
    Create summary statistics
    Parameters
    ----------
    df : Pandas DataFrame
        Data farme with evaluation results

    Returns:
    res : Pandas DataFrame
        A data frame with summary statistics
    """
    series = []
    for summary_func in [np.mean, np.std, np.median, np.min, np.max]:
        series.append(df.apply(summary_func))
    res = pd.concat(series, axis=1)
    res.columns = ['MEAN', 'SD', 'MEDIAN', 'MIN', 'MAX']
    return res


def compute_correlations_between_versions(df_old, df_new,
                                          human_score='sc1',
                                          id_column='spkitemid'):
    """
    Computes correlations in feature values between the two versions
    as well as the correlations between feature values and human score
    for each version
    Parameters
    ----------
    df_old : pandas DataFrame
        Data frame with feature values for the 'old' model

    df_new : pandas DataFrame
        Data frame with feature valeus for the 'new' model

    human_score : str
        Name of the column containing human score. Defaults to 'sc1'.
        Must be the same for both data sets.

    id_column : str
        Name of the column containing id for each response. Defaults to 'spkitemid'
        Must be the same for both data sets.

    Returns
    -------
    df_correlations: pandas DataFrame
        Data frame with a row for each feature and the following columns:
        N -  total N cases
        human_old - correlation with human score in the old version
        human_new - correlation with human score in the new version
        old_new - correlation between old and new version

    Raises
    ------
    ValueError
        If there are no shared features between the two sets or if there are 
        no share responses between the two sets
    """
    # Only use features that appear in both datasets

    features_old = [column for column in df_old if not column in [id_column, human_score]]
    features_new = [column for column in df_new if not column in [id_column, human_score]]
    features = list(set(features_old).intersection(features_new))

    if len(features) == 0:
        raise ValueError("There are no matching features in these two data sets.")

    columns = features + [id_column, human_score]

    # merge the two data sets and display a warning if there are non-matching
    # ids
    df_merged = pd.merge(df_old[columns], df_new[columns],
                         on=[id_column], suffixes=['%%%old', '%%%new'])

    if len(df_merged) == 0:
        raise ValueError("There are no shared ids between these two datasets")

    if len(df_merged) != len(df_old):
        warnings.warn("Some responses from the old data "
                      "were not present in the new data and therefore "
                      "were excluded from the analysis")
    if len(df_merged) != len(df_new):
        warnings.warn("Some responses from the new data "
                      "were not present in the old data and therefore "
                      "were excluded from the analysis")

    # compute correlations between each feature and human score.
    # we are using the same approach as used in analysis.py
    correlation_list = []
    for feature in features:
        # compute correlations
        df_cor = pd.DataFrame({'Feature': [feature],
                               'N': len(df_merged),
                               'human_old': pearsonr(df_merged['{}%%%old'.format(human_score)],
                                                     df_merged['{}%%%old'.format(feature)])[0],
                               'human_new': pearsonr(df_merged['{}%%%new'.format(human_score)],
                                                     df_merged['{}%%%new'.format(feature)])[0],
                               'old_new': pearsonr(df_merged['{}%%%new'.format(feature)],
                                                   df_merged['{}%%%old'.format(feature)])[0]})
        correlation_list.append(df_cor)

    df_correlations = pd.concat(correlation_list)
    df_correlations.index=df_correlations['Feature']
    return(df_correlations)


def process_confusion_matrix(conf_matrix):
    """
    Process confusion matrix to add 'human' and 'machine'
    to column names
    Parameters
    ----------
    confmatrix : pandas DataFrame
        pandas Data Frame containing the confusion matrix

    Returns:
    conf_matrix_renamed : Pandas DataFrame
        A data frame with confusion matrix and re-named indices and columns
    """
    conf_matrix_renamed = conf_matrix.copy()
    conf_matrix_renamed.index = ['machine {}'.format(n) for n in conf_matrix.index]
    conf_matrix_renamed.columns = ['human {}'.format(x) for x in conf_matrix.columns]
    return conf_matrix_renamed


def load_rsmtool_output(csvdir, figdir, experiment_id, prefix, groups_eval):

    """
    Read all outputs from the experiment. For each output we first check whether the file exists
    to allow comparing experiments with different sets of outputs.
    Parameters
    ----------
    csvdir : str
        path to the directory containing output .csv files
    figdir : str
        path to the directory containing output figures
    experiment_id : str
        original experiment_id used to generate the output files
    prefix: str
        must be set to 'scale' or 'raw'. Indicates whether the score
        is scaled or not
    groups_eval: list
        list of subgroups for subgroup evaluation

    Returns
    -------
    res : dict
        a dictionary with .csv outputs converted to Pandas Dataframe
        Default dictionary value is an empty data frame
    figs: dict
        a dictionary with figures
    """

    res = defaultdict(pd.DataFrame)
    figs = {}

    # feature distributions and the inter-feature correlations

    feature_train_file = join(csvdir, '{}_train_features.csv'.format(experiment_id))
    if exists(feature_train_file):
        res['df_train_features'] = pd.read_csv(feature_train_file)

    feature_distplots_file = join(figdir, '{}_distrib.svg'.format(experiment_id))
    if exists(feature_distplots_file):
        with open(feature_distplots_file, 'rb') as f:
            figs['feature_distplots'] = base64.b64encode(f.read()).decode('utf-8')

    feature_cors_file = join(csvdir, '{}_cors_processed.csv'.format(experiment_id))
    if exists(feature_cors_file):
        res['df_feature_cors'] = pd.read_csv(feature_cors_file, index_col=0)

    # df_scores
    scores_file = join(csvdir, '{}_pred_processed.csv'.format(experiment_id))
    if exists(scores_file):
        df_scores = pd.read_csv(scores_file, converters={'spkitemid': str})
        res['df_scores'] = df_scores[['spkitemid', 'sc1', prefix]]

    # model coefficients if present
    betas_file = join(csvdir, '{}_betas.csv'.format(experiment_id))
    if exists(betas_file):
        res['df_coef'] = pd.read_csv(betas_file, index_col=0)
        res['df_coef'].index.name = None

    # read in the model fit files if present
    model_fit_file = join(csvdir, '{}_model_fit.csv'.format(experiment_id))
    if exists(model_fit_file):
        res['df_model_fit'] = pd.read_csv(model_fit_file)

    # human human agreement
    consistency_file = join(csvdir, '{}_consistency.csv'.format(experiment_id))

    # load if consistency file is present
    if exists(consistency_file):
        df_consistency = pd.read_csv(consistency_file, index_col=0)
        res['df_consistency'] = df_consistency

    # degradation
    degradation_file = join(csvdir, "{}_degradation.csv".format(experiment_id))

    # load if degradation file is present
    if exists(degradation_file):
        df_degradation = pd.read_csv(degradation_file, index_col=0)
        res['df_degradation'] = df_degradation

    # use the raw columns or the scale columns depending on the prefix
    existing_eval_cols = _df_eval_columns_existing_raw if prefix == 'raw' else _df_eval_columns_existing_scale
    renamedict = raw_renamedict if prefix == 'raw' else scale_renamedict

    # read in the short version of the evaluation metrics for all data
    short_metrics_list = ["N", "Adj. Agmt.(br)", "Agmt.(br)", "K(br)", "Pearson(b)", "QWK(br)", "R2(b)", "RMSE(b)"]
    eval_file_short = join(csvdir, '{}_eval_short.csv'.format(experiment_id))
    if exists(eval_file_short):
        df_eval = pd.read_csv(eval_file_short, index_col=0)
        df_eval = df_eval[existing_eval_cols]
        df_eval = df_eval.rename(columns=renamedict)
        res['df_eval'] = df_eval[short_metrics_list]
        res['df_eval'].index.name = None

    eval_file = join(csvdir, '{}_eval.csv'.format(experiment_id))
    if exists(eval_file):
        res['df_eval_for_degradation'] = pd.read_csv(eval_file, index_col=0)

    # read in the evaluation metrics by subgroup, if we are asked to
    for group in groups_eval:
        group_eval_file = join(csvdir, '{}_eval_by_{}.csv'.format(experiment_id, group))
        if exists(group_eval_file):
            df_eval = pd.read_csv(group_eval_file, index_col=0)
            df_eval = df_eval[existing_eval_cols]
            df_eval = df_eval.rename(columns=renamedict)
            res['df_eval_by_{}'.format(group)] = df_eval[short_metrics_list]
            res['df_eval_by_{}'.format(group)].index.name = None
            res['df_eval_by_{}_overview'.format(group)] = make_summary_stat_df(res['df_eval_by_{}'.format(group)])

            # set the ordering of mean/SD/SMD statistics
            res['df_eval_by_{}_m_sd'.format(group)] = df_eval[['N', 'H1 mean', 'H1 SD', 'score mean(br)', 'score SD(br)', 'score mean(b)', 'score SD(b)', 'SMD(br)', 'SMD(b)']]
            res['df_eval_by_{}_m_sd'.format(group)].index.name = None

    # read in the partial correlations vs. score for all data
    pcor_score_file = join(csvdir, '{}_pcor_score_all_data.csv'.format(experiment_id))
    if exists(pcor_score_file):
        res['df_pcor_sc1'] = pd.read_csv(pcor_score_file, index_col=0)
        res['df_pcor_sc1_overview'] = make_summary_stat_df(res['df_pcor_sc1'])

    # read in the partial correlations by subgroups, if we are asked to
    for group in groups_eval:
        group_pcor_file = join(csvdir, '{}_pcor_score_by_{}.csv'.format(experiment_id, group))
        if exists(group_pcor_file):
            res['df_pcor_sc1_by_{}'.format(group)] = pd.read_csv(group_pcor_file, index_col=0)
            res['df_pcor_sc1_{}_overview'.format(group)] = make_summary_stat_df(res['df_pcor_sc1_by_{}'.format(group)])

    # read in the marginal correlations vs. score for all data
    mcor_score_file = join(csvdir, '{}_margcor_score_all_data.csv'.format(experiment_id))
    if exists(mcor_score_file):
        res['df_mcor_sc1'] = pd.read_csv(mcor_score_file, index_col=0)
        res['df_mcor_sc1_overview'] = make_summary_stat_df(res['df_mcor_sc1'])

    # read in the partial correlations by subgroups, if we are asked to
    for group in groups_eval:
        group_mcor_file = join(csvdir, '{}_margcor_score_by_{}.csv'.format(experiment_id, group))
        if exists(group_mcor_file):
            res['df_mcor_sc1_by_{}'.format(group)] = pd.read_csv(group_mcor_file, index_col=0)
            res['df_mcor_sc1_{}_overview'.format(group)] = make_summary_stat_df(res['df_mcor_sc1_by_{}'.format(group)])

    pca_file = join(csvdir, '{}_pca.csv'.format(experiment_id))
    if exists(pca_file):
        res['df_pca'] = pd.read_csv(pca_file, index_col=0)
        res['df_pcavar'] = pd.read_csv(join(csvdir, '{}_pcavar.csv'.format(experiment_id)), index_col=0)

    descriptives_file = join(csvdir, '{}_feature_descriptives.csv'.format(experiment_id))
    if exists(descriptives_file):
        # we read all files pertaining to the descriptive analysis together since we merge the outputs
        res['df_descriptives'] = pd.read_csv(descriptives_file, index_col=0)

        # this df contains only the number of features. this is used later for another two tables to show the number of features
        df_features_n_values = res['df_descriptives'][['N', 'min', 'max']]

        res['df_descriptives'] = res['df_descriptives'][['N', 'mean', 'std. dev.', 'skewness', 'kurtosis']]

        outliers_file = join(csvdir, '{}_feature_outliers.csv'.format(experiment_id))
        df_outliers = pd.read_csv(outliers_file, index_col=0)
        df_outliers = df_outliers.rename(columns={'upper': 'Upper',
                                                  'lower': 'Lower',
                                                  'both': 'Both',
                                                  'upperperc': 'Upper %',
                                                  'lowerperc': 'Lower %',
                                                  'bothperc': 'Both %'})
        df_outliers_columns = df_outliers.columns.tolist()
        res['df_outliers'] = df_outliers

        # join with df_features_n_values to get the value of N
        res['df_outliers'] = pd.merge(res['df_outliers'], df_features_n_values, left_index=True, right_index=True)[['N'] + df_outliers_columns]

        # join with df_features_n_values to get the value of N
        res['df_percentiles'] = pd.read_csv(join(csvdir, '{}_feature_descriptivesExtra.csv'.format(experiment_id)), index_col=0)
        res['df_percentiles'] = pd.merge(res['df_percentiles'], df_features_n_values, left_index=True, right_index=True)

        res['df_percentiles']["Mild outliers (%)"] = res['df_percentiles']["Mild outliers"]/res['df_percentiles']["N"].astype(float)*100
        res['df_percentiles']["Extreme outliers (%)"] = res['df_percentiles']["Extreme outliers"]/res['df_percentiles']["N"].astype(float)*100

        res['df_percentiles'] = res['df_percentiles'][['N', 'min', 'max', '1%', '5%', '25%', '50%', '75%', '95%', '99%', 'IQR', 'Mild outliers', 'Mild outliers (%)', 'Extreme outliers', 'Extreme outliers (%)']]

    confmatrix_file = join(csvdir, '{}_confMatrix.csv'.format(experiment_id))
    if exists(confmatrix_file):
        conf_matrix = pd.read_csv(confmatrix_file, index_col=0)
        res['df_confmatrix'] = process_confusion_matrix(conf_matrix)

    score_dist_file = join(csvdir, '{}_score_dist.csv'.format(experiment_id))
    if exists(score_dist_file):
        df_score_dist = pd.read_csv(score_dist_file, index_col=1)
        df_score_dist.rename(columns={'sys_{}'.format(prefix): 'sys'}, inplace=True)
        res['df_score_dist'] = df_score_dist[['human', 'sys', 'difference']]

    # read in the feature boxplots by subgroup, if we were asked to
    for group in groups_eval:
        feature_boxplot_prefix = join(figdir, '{}_feature_boxplot_by_{}'.format(experiment_id, group))
        svg_file = join(feature_boxplot_prefix + '.svg')
        png_file = join(feature_boxplot_prefix + '.png')
        if exists(svg_file):
            with open(svg_file, 'rb') as f:
                figs['feature_boxplots_by_{}_svg'.format(group)] = base64.b64encode(f.read()).decode('utf-8')
        elif exists(png_file):
            with open(png_file, 'rb') as f:
                figs['feature_boxplots_by_{}_png'.format(group)] = base64.b64encode(f.read()).decode('utf-8')

    # read in the betas image if exists
    betas_svg = join(figdir, '{}_betas.svg'.format(experiment_id))
    if exists(betas_svg):
        with open(betas_svg, 'rb') as f:
            figs['betas'] = base64.b64encode(f.read()).decode('utf-8')

    # read in the evaluation barplots by subgroup, if we were asked to
    for group in groups_eval:
        eval_barplot_svg_file = join(figdir, '{}_eval_by_{}.svg'.format(experiment_id, group))
        if exists(eval_barplot_svg_file):
            with open(eval_barplot_svg_file, 'rb') as f:
                    figs['eval_barplot_by_{}'.format(group)] = base64.b64encode(f.read()).decode('utf-8')

    pca_svg_file = join(figdir, '{}_pca.svg'.format(experiment_id))
    if exists(pca_svg_file):
        with open(pca_svg_file, 'rb') as f:
            figs['pca_scree_plot'] = base64.b64encode(f.read()).decode('utf-8')

    return (res, figs)
