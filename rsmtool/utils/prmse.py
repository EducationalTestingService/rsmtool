"""
PRMSE utilities.

Utility classes and functions related to computing test
theory based evaluations.

The derivations and formulas were provided by Matt Johnson.

:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

import warnings

import numpy as np
import pandas as pd


def get_n_human_scores(human_scores):
    """
    Get the number of available human scores for each response.

    Parameters
    ----------
    human_scores : array-like of shape (n_samples, n_ratings)
        Human ratings for each response.

    Returns
    -------
    n_scores : array-like of shape (n_samples, )
        Total number of not None human scores
    """
    n_scores = (~np.isnan(human_scores)).sum(axis=1)
    return n_scores


def variance_of_errors(human_scores):
    """
    Estimate the variance of errors in human scores.

    Parameters
    ----------
    human_scores : array-like of shape (n_samples, n_ratings)
        Human ratings for each response.

    Returns
    -------
    variance_of_errors : float
        Estimated variance of errors in human scores.
    """
    # we first compute the total number of scores
    # available for each response

    n_scores = get_n_human_scores(human_scores)

    # we will only be using responses with more
    # than one score
    multiple_mask = n_scores > 1

    # show a warning and return None
    # if we don't have valid human scores
    if multiple_mask.sum() == 0:
        warnings.warn(
            "True score evaluations cannot be "
            "computed because none of the responses in the "
            "evaluation set has valid "
            "system scores and 2 human scores."
        )
        return None

    else:
        # only select the responses with multiple scores
        multiple_scores = human_scores[multiple_mask]

        n_scores = n_scores[multiple_mask]

        # now let's compute the rater error variance for each
        # response
        response_variances = np.nanvar(multiple_scores, ddof=1, axis=1)

        # finally, let's compute the variance of errors as a weighted average
        # of response variances

        variance_of_errors = np.average(response_variances, weights=n_scores - 1)

        return variance_of_errors


def true_score_variance(human_scores, variance_errors_human=None):
    """
    Compute variance of true scores for multiple raters.

    Parameters
    ----------
    human_scores : array-like of shape (n_samples, n_ratings)
        Human ratings for each response.

    variance_errors_human : float, optional
        Estimated variance of errors in human scores.
        If ``None``, the variance will be estimated
        from the data. In this case at least some responses
        must have more than one human score.
        Defaults to ``None``.


    Returns
    -------
    variance_true_scores : float
        Variance of true scores.
    """
    # if we don't have variance of errors, compute it
    # from the data

    if variance_errors_human is None:
        variance_errors_human = variance_of_errors(human_scores)

    # if it's still None, return None
    if variance_errors_human is None:
        return None

    else:
        # compute mean human score and total number of scores
        # for each response
        mean_scores = np.nanmean(human_scores, axis=1)
        n_scores = get_n_human_scores(human_scores)

        # compute overall mean
        mean_human_score = np.nanmean(human_scores)

        # let N be total number of responses
        N = len(human_scores)

        # let M be total number of human ratings
        M = n_scores.sum()

        # compute squared deviations
        squared_devs = (mean_scores - mean_human_score) ** 2

        # adjust them by the number of human scores available
        # for each responses: deviations with higher number of
        # human scores are assigned a greater weight
        adjusted_squared_devs = n_scores * squared_devs

        # compute sum of squares
        sum_of_squares = adjusted_squared_devs.sum()

        # now compute the numerator as sum of squares
        # adjusted for the variance of human errors
        numerator = sum_of_squares - (N - 1) * variance_errors_human

        # compute the denominator as the adjusted total number of scores
        denominator = M - ((n_scores**2).sum() / M)

        # finally compute variance of true scores
        variance_true_scores = numerator / denominator

        return variance_true_scores


def mse_true(system, human_scores, variance_errors_human=None):
    """
    Compute mean squared error (MSE) when predicting true score from system score.

    Parameters
    ----------
    system : array-like of shape (n_samples,)
        System scores for each response.
    human_scores : array-like of shape (n_samples, n_ratings)
        Human ratings for each response.
    variance_errors_human : float, optional
        Estimated variance of errors in human scores.
        If ``None``, the variance will be estimated from
        the data. In this case at least some responses must
        have more than one human score.
        Defaults to ``None``.

    Returns
    -------
    variance_true_scores : float
        Variance of true scores.
    """
    # if we don't have variance of errors, compute it
    # from the data

    if variance_errors_human is None:
        variance_errors_human = variance_of_errors(human_scores)

    # if it's still None, return None
    if variance_errors_human is None:
        return None

    else:
        # get total number of scores for each response
        n_scores = get_n_human_scores(human_scores)
        mean_scores = np.nanmean(human_scores, axis=1)

        N = len(system)

        se = ((mean_scores - system) ** 2) * n_scores

        # Compute mean squared error when predicting true score
        mse = (se.sum() - N * variance_errors_human) / n_scores.sum()
    return mse


def prmse_true(system, human_scores, variance_errors_human=None):
    """
    Compute PRMSE when predicting true score from system scores.

    PRMSE = Proportional Reduction in Mean Squared Error.
    The formula to compute PRMSE implemented in RSMTool
    was derived at ETS by Matthew S. Johnson. See
    `Loukina et al. (2020) <https://aclanthology.org/2020.bea-1.2.pdf>`_
    for further information about PRMSE.

    Parameters
    ----------
    system : array-like of shape (n_samples,)
        System scores for each response.
    human_scores : array-like of shape (n_samples, n_ratings)
        Human ratings for each response.
    variance_errors_human : float, optional
        Estimated variance of errors in human scores.
        If ``None``, the variance will be estimated from
        the data. In this case at least some responses must
        have more than one human score.
        Defaults to ``None``.

    Returns
    -------
    prmse : float
        Proportional reduction in mean squared error
    """
    # check that human_scors is a two dimensional array
    # and reshape if necessary
    if len(human_scores.shape) == 1:
        current_length = human_scores.shape[0]
        # first assume we have a pandas series
        try:
            human_scores = human_scores.values.reshape(current_length, 1)
        # if not, treat this as an array
        except AttributeError:
            human_scores = human_scores.reshape(current_length, 1)

    if variance_errors_human is None:
        variance_errors_human = variance_of_errors(human_scores)

    # if it's still None, return None
    if variance_errors_human is None:
        return None

    else:
        variance_true = true_score_variance(human_scores, variance_errors_human)

        mse = mse_true(system, human_scores, variance_errors_human)

        prmse = 1 - (mse / variance_true)

        return prmse


def get_true_score_evaluations(
    df, system_score_columns, human_score_columns, variance_errors_human=None
):
    """
    Generate true score evaluations for HTML reports.

    Parameters
    ----------
    df: pandas DataFrame
        Input data frame. Must contain columns listed in
        ``system_score_columns`` and ``human_score_columns``.
    system_score_columns: str or list
        System score column name or list of columns containing system scores.
    human_score_columns: str or list
        Human score column or list of columns containing human scores.
        True score evaluations require estimating variance of human errors,
        which can only be computed when a subset of responses has
        two or more human ratings. If  ``human_score_columns`` is
        a single column name,  ``variance_errors_human`` must also
        be specified.
    variance_errors_human : float, optional
        Estimated variance of errors in human scores.
        If ``None``, the variance will be estimated from
        the data in which case some responses must have more
        than one human rating.
        Defaults to ``None``.

    Returns
    -------
    prmse_metrics: pandas DataFrame
        DataFrame containing different evaluation metrics related to the evaluation
        of system scores against true scores. The column names are:

        - "N": total number of responses
        - "N raters": maximum number of ratings available for a single response
        - "N_single": total number of responses with a single human score
        - "N_multiple": total number of responses with more than one
          human score
        - "variance_of_errors": estimated variance of human errors
        - "tru_var": estimated true score variance
        - "mse_true": mean squared error when predicting true score from
          machine score
        - "prmse": proportional reduction in mean squared error when
          predicting true score
    """
    # check that if we only have one human column, we were also given
    # variance of errors
    if isinstance(human_score_columns, str):
        if variance_errors_human is None:
            raise (
                ValueError(
                    "True score evaluations require estimating "
                    "variance of human errors, "
                    "which can only be computed when a subset "
                    "of responses has two or more human ratings. "
                    "If a single human_score_column "
                    "is supplied, one must also specify variance_errors_human"
                )
            )

    if isinstance(system_score_columns, str):
        system_score_columns = [system_score_columns]

    # compute variance of errors if it wasn't specified
    if variance_errors_human is None:
        variance_errors_human = variance_of_errors(df[human_score_columns])

    # compute prmse
    prmse_all = []
    for system in system_score_columns:
        mse = mse_true(df[system], df[human_score_columns], variance_errors_human)
        prmse = prmse_true(df[system], df[human_score_columns], variance_errors_human)
        prmse_metrics = pd.Series({"MSE true": mse, "PRMSE true": prmse}, name=system)
        prmse_all.append(prmse_metrics)

    df_prmse = pd.concat(prmse_all, axis=1, sort=True).transpose()

    score_counts = get_n_human_scores(df[human_score_columns])

    # compute values that are the same for all scores
    df_prmse.insert(0, "N", len(df))
    df_prmse.insert(1, "N raters", score_counts.max())
    df_prmse.insert(2, "N single", (score_counts == 1).sum()),
    df_prmse.insert(3, "N multiple", (score_counts > 1).sum()),
    df_prmse.insert(4, "Variance of errors", variance_errors_human)
    df_prmse.insert(
        5,
        "True score var",
        true_score_variance(df[human_score_columns], variance_errors_human),
    )
    return df_prmse
