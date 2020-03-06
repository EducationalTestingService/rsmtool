"""
Utility classes and functions related to computing test
theory based evaluations

:author: Anastassia Loukina (aloukina@ets.org)
:author: Jeremy Biggs (jbiggs@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

import pandas as pd

import numpy as np

from scipy.special import comb


def get_n_human_scores(df_human_scores):
    """
    Compute number of human scores for each response
    Parameters
    ----------
    df_human_scores : pandas DataFrame
        DataFrame containing columns with human scores only

    Returns
    -------
    n_scores : pandas Series
        Pandas series with total number of not None human scores
        for each row in df_human_scores
    """
    score_mask = ~df_human_scores.isnull()
    n_scores = score_mask.sum(axis=1)
    return n_scores



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

def variance_of_errors(df_human_scores):
    '''Compute variance of human errors.

    Parameters
    ----------
    df_human_scores : pandas DataFrame
        DataFrame containing columns with human scores only

    Returns
    variance_of_errors : float
        Estimated variance of human errors
    '''

    # we only use responses with more than 1 score

    n_scores = get_n_human_scores(df_human_scores)

    df_multiple = df_human_scores[n_scores>1]

    # raise an error if we don't have any such responses
    if len(df_multiple) == 0:
        raise ValueError("Variance of human errors "
                         "necessary for true score "
                         "evaluations requires "
                         "at least a subset of responses "
                         "to be scored by 2 or more "
                         "raters.")


    # we convert the dataframe to an array since
    # in these computations having column names
    # interferes with the computation

    score_matrix = df_multiple.to_numpy()

    # we next calculate the difference between ratings
    # for each response

    def compute_difference(ratings):
        total_ratings = len(ratings)
        ratings = ratings[~np.isnan(ratings)]
        n = len(ratings)
        # Difference has dimensions (n-1,)
        difference = ratings[1:] - ratings[:-1].cumsum()/(np.arange(1, n))
        # Compute multiplication factor.
        # This also has dimension (n-1,)
        factor = np.arange(1, n)/np.arange(2, n+1)
        # Compute contrast. This also has dimensions n-1
        contrast = np.sqrt(factor)*difference
        # now we need to pad it back to the total number of
        # original ratings
        pad_width = total_ratings - n
        contrast = np.pad(contrast,
                          (0, pad_width),
                          constant_values=np.nan)
        return contrast

    differences = np.apply_along_axis(compute_difference, 1, score_matrix)

    # variance of errors is the mean of squared differences
    variance_of_errors = np.nanmean((differences**2))

    return variance_of_errors


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
       Average human scores for double-scored responses
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
       Average human scores for double-scored responses
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
        Average human scores used to compute the variance
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



def true_score_variance(df_human_scores,
                        variance_errors_human=None):

    """
    Compute variance of true scores
    for multiple raters

    Parameters
    ----------
    df_human_scores : pandas DataFrame
        Data frame with human scores used to compute the variance
        Each score should be stored in a separate column
    variance_errors_human : float
        Estimated variance of errors in human scores
        When no value is supplied, the variance will
        be estimated from the data. In this case
        at least some responses must have more than
        one human score.


    Returns
    -------
    variance_true_scores : float
        Variance of true scores
    """

    # if we don't have variance of errors, compute it
    # from the data

    if not variance_errors_human:
        variance_errors_human = variance_of_errors(df_human_scores)

    # compute mean human score and total number of scores
    # for each response
    mean_scores = df_human_scores.mean(axis=1)
    n_scores = get_n_human_scores(df_human_scores)

    # compute overall mean
    mean_human_score = np.nanmean(df_human_scores.values)

    # let N be total number of responses
    N = len(df_human_scores)

    # let M be total number of human ratings
    M = n_scores.sum()

    # compute squared deviations
    squared_devs = (mean_scores - mean_human_score)**2

    adjusted_squared_devs = n_scores * squared_devs

    sum_of_squares = adjusted_squared_devs.sum()

    numerator = sum_of_squares - (N-1)*variance_errors_human

    denominator = M - ((n_scores**2).sum()/M)

    variance_true_scores = numerator/denominator

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



def mse_true(system,
             df_human_scores,
             variance_errors_human=None):

    """
    Compute mean square error (MSE) when predicting true score
    from system score.

    Parameters
    ----------
    system : pandas Series
        System scores
    df_human_scores : pandas DataFrame
        Data frame with human scores used to compute the variance
        Each score should be stored in a separate column
    variance_errors_human : float
        Estimated variance of errors in human scores
        When no value is supplied, the variance will
        be estimated from the data. In this case
        at least some responses must have more than
        one human score.


    Returns
    -------
    variance_true_scores : float
        Variance of true scores
    """

    # if we don't have variance of errors, compute it
    # from the data

    if not variance_errors_human:
        variance_errors_human = variance_of_errors(df_human_scores)


    # get total number of scores for each response
    n_scores = get_n_human_scores(df_human_scores)
    mean_scores = df_human_scores.mean(axis=1)

    N = len(system)

    se = ((mean_scores - system)**2) * n_scores

    # Compute mean squared error when predicting true score
    mse = (se.sum() - N * variance_errors_human) / n_scores.sum()
    return mse



def prmse_true(system,
               df_human_scores,
               variance_errors_human=None):
    """
    Compute Proportional Reduction in Mean Squared Error (PRMSE)
    when predicting true score from system scores.

    Parameters
    ----------
    system : pandas Series
        System scores
    df_human_scores : pandas DataFrame
        Data frame with human scores used to compute the variance
        Each score should be stored in a separate column
    variance_errors_human : float
        Estimated variance of errors in human scores
        When no value is supplied, the variance will
        be estimated from the data. In this case
        at least some responses must have more than
        one human score.

    Returns
    -------
    prmse : float
        Proportional reduction in mean square error
    """

    if not variance_errors_human:
        variance_errors_human = variance_of_errors(df_human_scores)

    variance_true = true_score_variance(df_human_scores, variance_errors_human)

    mse = mse_true(system, df_human_scores, variance_errors_human)

    prmse = 1 - (mse / variance_true)

    return prmse



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
