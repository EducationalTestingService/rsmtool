"""
Utility classes and functions related to computing test
theory based evaluations

:author: Anastassia Loukina (aloukina@ets.org)
:author: Jeremy Biggs (jbiggs@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

import pandas as pd


def compute_variance_of_errors(df,
                               h1_column='sc1',
                               h2_column='sc2',):
    """
    Compute variance of errors in human scores.

    Parameters
    ----------
    df : pandas DataFrame
        Input data frame. Must contain columns `h1_column` and `h2_column`.
    h1_column : str, optional
        The first human score column name.
        Defaults to 'sc1'.
    h2_column : str, optional
        The second human score column name.
        Defaults to 'sc2'.

    Returns
    -------
    variance_errors_human: float
         Variance of errors in human scores

    Raises
    ------
    ValueError
         If any of the values in `h2_column` or `h1_column` are NaN.
    """

    # check that all responses are double scored
    if df[h2_column].isnull().any() or df[h1_column].isnull().any():
        raise ValueError("Variance of errors should only be computed on double-scored responses.")

    N = len(df)

    # estimate variance of errors in human scores for double-scored responses
    # as 1/2 of average squared difference between the two scores
    variance_errors_human = 1 / (2 * N) * ((df[h1_column] - df[h2_column])**2).sum()

    return variance_errors_human


def compute_true_score_var_subset_double_scored(single_human_scores,
                                                double_human_scores,
                                                variance_errors_human):
    """
    Compute variance of true scores
    in a situation where only some
    of the responses are double scored.

    Parameters
    ----------
    single_human_scores : pandas Series
        Human scores for single-scored responses
    double_human_scores : pandas Series
       Human scores for double-scored responses
    variance_errors_human : float
        Estimated variance of errors in human scores

    Returns
    -------
    variance_true_scores : float
        Variance of true scores
    """

    # N of examinees with 1 human score.
    n_1 = len(single_human_scores)

    # N of examinees with 2 human scores.
    n_2 = len(double_human_scores)

    N = n_1 + n_2

    # compute squared deviation of sc_bar score for each responses
    # from mean sc_bar score.
    # This is then multiplied by a coefficient corresponding to the number
    # of human scores (1 for single scores responses, 2 for double-scored responses)
    # so that the responses with higher number of human scores are
    # weighed higher

    sc_bar_mean = pd.concat([single_human_scores, double_human_scores]).mean()

    squared_dist_single = (single_human_scores - sc_bar_mean)**2
    squared_dist_double = 2 * (double_human_scores - sc_bar_mean)**2

    # concatenate both data frames
    squared_dist = pd.concat([squared_dist_single, squared_dist_double], sort=True)

    # third, compute variance of true scores.

    # the numerator corresponds to the weighed sum of squares adjusted for
    # the estimated variance or errors in human scores
    numerator = (squared_dist.sum() - (N - 1) * variance_errors_human)

    # the denominator is construed to ensure correct treatment of
    # all cases regardless of what percentage responses is double-scored
    denominator = (N - 1) + (n_2 * (n_1 + 2 * n_2 - 2)) / (n_1 + 2 * n_2)

    variance_true_scores = numerator / denominator

    return variance_true_scores


def compute_mse_subset_double_scored(single_human_scores,
                                     double_human_scores,
                                     single_system_scores,
                                     double_system_scores,
                                     variance_errors_human):
    """
    Compute MSE when predicting true score from system scores
    in a situation where only some
    of the responses are double scored.

    Parameters
    ----------
    single_human_scores : pandas Series
        Human scores for single-scored responses
    double_human_scores : pandas Series
       Human scores for double-scored responses
    single_system_scores : pandas Series
        System scores for single-scored responses
    double_human_scores : pandas Series
        System scores for double-scored responses
    variance_errors_human : float
        Variance of errors in human scores

    Returns
    -------
    mse : float
       Mean squared error
    """

    # N of examinees with 1 human score.
    n_1 = len(single_human_scores)

    # N of examinees with 2 human scores.
    n_2 = len(double_human_scores)

    N = n_1 + n_2

    # compute squared error c_i*(sc_bar_i - system_i)**2.
    # Note that for double-scored responses c = 2.
    se_single = (single_human_scores - single_system_scores)**2
    se_double = 2 * (double_human_scores - double_system_scores)**2

    # concatenate both sets
    se = pd.concat([se_single, se_double], sort=True)

    # Compute mean squared error when predicting true score
    mse = (se.sum() - N * variance_errors_human) / (n_1 + 2 * n_2)

    return mse


def compute_true_score_var_all_double_scored(human_scores,
                                             variance_errors_human):
    """
    Compute variance of true scores
    in a situation where all
    responses are double scored.

    Parameters
    ----------
    human_scores : pandas Series
        Human scores used to compute the variance
    variance_errors_human : float
        Variance of errors in human scores

    Returns
    -------
    variance_true_scores : float
        Variance of true scores
    """
    N = len(human_scores)

    # compute variance of observed human scores:
    variance_observed_scores = ((human_scores - human_scores.mean())**2).sum() / (N - 1)

    # compute variance of true scores as variance of observed
    # scores adjusted for estimated variance of errors in human scores
    variance_true_scores = variance_observed_scores - variance_errors_human / 2

    return variance_true_scores


def compute_mse_all_double_scored(human_scores,
                                  system_scores,
                                  variance_errors_human):
    """
    Compute MSE when predicting true score from system scores
    in a situation where all
    of the responses are double scored.

    Parameters
    ----------
    human_scores : pandas Series
        Human scores
    system_scores : pandas Series
        System scores
    variance_errors_human : float
        Variance of errors in human scores

    Returns
    -------
    mse : float
        Mean squared error
    """

    N = len(human_scores)

    # compute MSE when predicting true score from system score
    # as MSE for observed scores adjusted for estimated variance of errors
    # in human scores
    mse = ((human_scores - system_scores)**2).sum() / N - variance_errors_human / 2

    return mse


def compute_prmse(df,
                  system_score_columns,
                  h1_column='sc1',
                  h2_column='sc2',
                  ddof=1):
    """
    Compute Proportional Reduction in Mean Squared Error (PRMSE)
    when predicting true score from system scores.

    Parameters
    ----------
    df: pandas DataFrame
        Input data frame. Must contain columns `sc1`, `sc2` and the columns
        `listed in system_score_columns`.
    system_score_columns: str or list
        System score column name or list of columns containing system scores
    h1_column : str, optional
        The first human score column name.
        Defaults to 'sc1'.
    h2_column : str, optional
        The second human score column name.
        Defaults to 'sc2'.
    ddof : int, optional
        Means Delta Degrees of Freedom. The divisor used in
        calculations is N - ddof, where N represents the
        number of elements.
        Defaults to 1.

    Returns
    -------
    prmse_metrics: pandas DataFrame
        DataFrame containing different evaluation metrics related to the evaluation
        of system scores against true scores:
        - `N`: total number of responses
        - `N_single`: total number of responses with a single human score
        - `N_double`: total number of responses with two human scores
        - `h1_var_single`: variance of first human score for single-scored responses
        - `h1_var_double`: variance of first human score for double-scored responses
        - `h2_var_double`: variance of second human score for double-scored responses
        - `tru_var`: estimated true score variance
        - `system_var_all`:  variance of system scores for all responses
        - `system_var_double`: variance of system scores for double-scored responses
        - `mse_true`: mean squared error when predicting true score from machine score
        - `prmse`: proportional reduction in mean squared error when predicting true score
    """
    if isinstance(system_score_columns, str):
        system_score_columns = [system_score_columns]

    # Split the data into single-scored and double-scored responses
    score_mask = df[h2_column].isnull()

    df_single = df[score_mask].copy()
    df_double = df[~score_mask].copy()

    # compute variance of errors
    variance_errors_human = compute_variance_of_errors(df_double,
                                                       h1_column,
                                                       h2_column)

    # compute average score for double-scored responses
    df_double['sc_bar'] = (df_double[h1_column] + df_double[h2_column]) / 2

    # compute variance of true scores
    if len(df_single) > 0:
        variance_true_scores = compute_true_score_var_subset_double_scored(df_single[h1_column],
                                                                           df_double['sc_bar'],
                                                                           variance_errors_human)

    else:
        variance_true_scores = compute_true_score_var_all_double_scored(df_double['sc_bar'],
                                                                        variance_errors_human)

    # compute MSE for each type of score
    prmse_all = []
    for system in system_score_columns:
        if len(df_single) > 0:
            mse = compute_mse_subset_double_scored(df_single[h1_column],
                                                   df_double['sc_bar'],
                                                   df_single[system],
                                                   df_double[system],
                                                   variance_errors_human)
        else:
            mse = compute_mse_all_double_scored(df_double['sc_bar'],
                                                df_double[system],
                                                variance_errors_human)

        prmse_metrics = pd.Series({'sys_var_single': df_single[system].var(ddof=ddof),
                                   'sys_var_double': df_double[system].var(ddof=ddof),
                                   'mse_true': mse,
                                   'prmse_true': 1 - mse / variance_true_scores}, name=system)
        prmse_all.append(prmse_metrics)

    # combine all results together
    df_prmse = pd.concat(prmse_all, axis=1, sort=True).transpose()

    # add numbers that are the same for all types of scores
    df_prmse.insert(0, 'N', len(df))
    df_prmse.insert(1, 'N_single', len(df_single))
    df_prmse.insert(2, 'N_double', len(df_double))
    df_prmse.insert(3, 'h1_var_single', df_single[h1_column].var(ddof=ddof))
    df_prmse.insert(4, 'h1_var_double', df_double[h1_column].var(ddof=ddof))
    df_prmse.insert(5, 'h2_var_double', df_double[h2_column].var(ddof=ddof))
    df_prmse.insert(6, 'true_var', variance_true_scores)

    return df_prmse
