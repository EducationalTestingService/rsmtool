"""
Utility classes and functions related to computing test
theory based evaluations.

The derivations and formulas were provided by Matt Johnson.

:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

import pandas as pd
import numpy as np


def get_n_human_scores(human_scores):
    """
    Compute number of human scores for each response
    Parameters
    ----------
    human_scores : array-like of shape (n_samples, n_ratings)
        Human ratings for each response

    Returns
    -------
    n_scores : array-like of shape (n-samples)
        Total number of not None human scores
    """
    n_scores = (~np.isnan(human_scores)).sum(axis=1)
    return n_scores


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


