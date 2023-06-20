"""
Utility functions for computing various RSMTool metrics.

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

import warnings

import numpy as np
import pandas as pd


def compute_expected_scores_from_model(model, featureset, min_score, max_score):
    """
    Compute expected scores using probability distributions over labels.

    This function only works with SKLL models.

    Parameters
    ----------
    model : skll.learner.Learner
        The SKLL learner object to use for computing the expected scores.
    featureset : skll.data.FeatureSet
        The SKLL featureset object for which predictions are to be made.
    min_score : int
        Minimum score level to be used for computing expected scores.
    max_score : int
        Maximum score level to be used for computing expected scores.

    Returns
    -------
    expected_scores: np.array
        A numpy array containing the expected scores.

    Raises
    ------
    ValueError
        If the given model cannot predict probability distributions.
    ValueError
        If the score range specified by ``min_score`` and ``max_score``
        does not match what the model predicts in its probability
        distribution.
    """
    if hasattr(model.model, "predict_proba"):
        # Tell the model we want probabiltiies as output. This is likely already set
        # to True but it might not be, e.g., when using rsmpredict.
        model.probability = True
        probability_distributions = model.predict(featureset, class_labels=False)
        # check to make sure that the number of labels in the probability
        # distributions matches the number of score points we have
        num_score_points_specified = max_score - min_score + 1
        num_score_points_in_learner = probability_distributions.shape[1]
        if num_score_points_specified != num_score_points_in_learner:
            raise ValueError(
                f"The specified number of score points ({num_score_points_specified}) "
                f"does not match that from the learner ({num_score_points_in_learner})."
            )
        expected_scores = probability_distributions.dot(range(min_score, max_score + 1))
    else:
        if model.model_type.__name__ == "SVC":
            raise ValueError(
                "Expected scores cannot be computed since the SVC model was "
                "not originally trained to predict probabilities."
            )
        else:
            raise ValueError(
                f"Expected scores cannot be computed since "
                f"{model.model_type.__name__} is not a probabilistic classifier."
            )

    return expected_scores


def covariance_to_correlation(m):
    """
    Implement the R ``cov2cor`` function in Python.

    Parameters
    ----------
    m : np.array
        The covariance matrix.

    Returns
    -------
    retval : np.array
        The cross-correlation matrix.

    Raises
    ------
    ValueError
        If the input matrix is not square.
    """
    # make sure the matrix is square
    numrows, numcols = m.shape
    if not numrows == numcols:
        raise ValueError("Input matrix must be square")

    Is = np.sqrt(1 / np.diag(m))
    retval = Is * m * np.repeat(Is, numrows).reshape(numrows, numrows)
    np.fill_diagonal(retval, 1.0)
    return retval


def partial_correlations(df):
    """
    Implement the R ``pcor`` function from ``ppcor`` package in Python.

    This computes partial correlations of each pair of variables
    in the given data frame ``df``, excluding all other variables.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame containing the feature values.

    Returns
    -------
    df_pcor : pd.DataFrame
        Data frame containing the partial correlations of of each
        pair of variables in the given data frame ``df``,
        excluding all other variables.
    """
    numrows, numcols = df.shape
    df_cov = df.cov()
    columns = df_cov.columns

    # return a matrix of nans if the number of columns is
    # greater than the number of rows. When the ncol == nrows
    # we get the degenerate matrix with 1 only. It is not meaningful
    # to compute partial correlations when ncol > nrows.

    # create empty array for when we cannot compute the
    # matrix inversion
    empty_array = np.empty((len(columns), len(columns)))
    empty_array[:] = np.nan
    if numcols > numrows:
        icvx = empty_array
    else:
        # if the determinant is less than the lowest representable
        # 32 bit integer, then we use the pseudo-inverse;
        # otherwise, use the inverse; if a linear algebra error
        # occurs, then we just set the matrix to empty
        try:
            assert np.linalg.det(df_cov) > np.finfo(np.float32).eps
            icvx = np.linalg.inv(df_cov)
        except AssertionError:
            icvx = np.linalg.pinv(df_cov)
            warnings.warn(
                "When computing partial correlations "
                "the inverse of the variance-covariance matrix "
                "was calculated using the Moore-Penrose generalized "
                "matrix inversion, due to its determinant being at "
                "or very close to zero."
            )
        except np.linalg.LinAlgError:
            icvx = empty_array

    pcor = -1 * covariance_to_correlation(icvx)
    np.fill_diagonal(pcor, 1.0)
    df_pcor = pd.DataFrame(pcor, columns=columns, index=columns)
    return df_pcor


def agreement(score1, score2, tolerance=0):
    """
    Compute the agreement between two raters, under given tolerance.

    Parameters
    ----------
    score1 : list of int
        List of rater 1 scores
    score2 : list of int
        List of rater 2 scores
    tolerance : int, optional
        Difference in scores that is acceptable.
        Defaults to 0.

    Returns
    -------
    agreement_value : float
        The percentage agreement between the two scores.
    """
    # make sure the two sets of scores
    # are for the same number of items
    assert len(score1) == len(score2)

    num_agreements = sum([int(abs(s1 - s2) <= tolerance) for s1, s2 in zip(score1, score2)])

    agreement_value = (float(num_agreements) / len(score1)) * 100
    return agreement_value


def standardized_mean_difference(
    y_true_observed,
    y_pred,
    population_y_true_observed_sd=None,
    population_y_pred_sd=None,
    method="unpooled",
    ddof=1,
):
    """
    Compute the standardized mean difference between system and human scores.

    The numerator is calculated as mean(y_pred) - mean(y_true_observed)
    for all of the available methods.

    Parameters
    ----------
    y_true_observed : array-like
        The observed scores for the group or subgroup.
    y_pred : array-like
        The predicted score for the group or subgroup.
    population_y_true_observed_sd : float, optional
        The population true score standard deviation.
        When the SMD is being calculated for a subgroup,
        this should be the standard deviation for the whole
        population.
        Defaults to ``None``.
    population_y_pred_sd : float, optional
        The predicted score standard deviation.
        When the SMD is being calculated for a subgroup,
        this should be the standard deviation for the whole
        population.
        Defaults to ``None``.
    method : str, optional
        The SMD method to use. Possible options are:

        - "williamson": Denominator is the pooled population standard deviation
          of ``y_true_observed`` and ``y_pred`` computed using
          ``population_y_true_observed_sd`` and ``population_y_pred_sd``.
        - "johnson": Denominator is ``population_y_true_observed_sd``.
        - "pooled": Denominator is the pooled standard deviation of
          ``y_true_observed`` and ``y_pred`` for this group.
        - "unpooled": Denominator is the standard deviation of
          ``y_true_observed`` for this group.

        Defaults to "unpooled".

    ddof : int, optional
        The delta degrees of freedom. The divisor used in
        calculations is N - ddof, where N represents the
        number of elements.
        Defaults to 1.

    Returns
    -------
    smd : float
        The SMD for the given group or subgroup.

    Raises
    ------
    ValueError
        If method is "williamson" and either ``population_y_true_observed_sd``
        or ``population_y_pred_sd`` is ``None``.
    ValueError
        If method is "johnson" and ``population_y_true_observed_sd`` is ``None``.
    ValueError
        If method is not one of {"unpooled", "pooled", "williamson", "johnson"}.

    Note
    ----
    - The "williamson" implementation was recommended by Williamson,
      et al. (2012).
    - The metric is only applicable when both sets of scores are on
      the same scale.
    """
    numerator = np.mean(y_pred) - np.mean(y_true_observed)

    method = method.lower()
    if method == "unpooled":
        denominator = np.std(y_true_observed, ddof=ddof)
    elif method == "pooled":
        denominator = np.sqrt(
            (np.std(y_true_observed, ddof=ddof) ** 2 + np.std(y_pred, ddof=ddof) ** 2) / 2
        )
    elif method == "johnson":
        if population_y_true_observed_sd is None:
            raise ValueError(
                "If `method='johnson'`, then `population_y_true_observed_sd` " "must be provided."
            )
        denominator = population_y_true_observed_sd
    elif method == "williamson":
        if population_y_true_observed_sd is None or population_y_pred_sd is None:
            raise ValueError(
                "If `method='williamson'`, both `population_y_true_observed_sd` "
                "and `population_y_pred_sd` must be provided."
            )
        denominator = np.sqrt((population_y_true_observed_sd**2 + population_y_pred_sd**2) / 2)
    else:
        possible_methods = {"'unpooled'", "'pooled'", "'johnson'", "'williamson'"}
        raise ValueError(
            f"The available methods are {{{', '.join(possible_methods)}}}; you selected {method}."
        )

    # if the denominator is zero, then return NaN as the SMD
    smd = np.nan if denominator == 0 else numerator / denominator
    return smd


def difference_of_standardized_means(
    y_true_observed,
    y_pred,
    population_y_true_observed_mn=None,
    population_y_pred_mn=None,
    population_y_true_observed_sd=None,
    population_y_pred_sd=None,
    ddof=1,
):
    """
    Calculate the difference between standardized means.

    First, standardize both observed and predicted scores to z-scores using
    mean and standard deviation for the whole population. Then
    calculate differences between standardized means for each subgroup.

    Parameters
    ----------
    y_true_observed : array-like
        The observed scores for the group or subgroup.
    y_pred : array-like
        The predicted score for the group or subgroup.
        The predicted scores.
    population_y_true_observed_mn : float, optional
        The population true score mean.
        When the DSM is being calculated for a subgroup,
        this should be the mean for the whole
        population.
        Defaults to ``None``.
    population_y_pred_mn : float, optional
        The predicted score mean.
        When the DSM is being calculated for a subgroup,
        this should be the mean for the whole
        population.
        Defaults to ``None``.
    population_y_true_observed_sd : float, optional
        The population true score standard deviation.
        When the DSM is being calculated for a subgroup,
        this should be the standard deviation for the whole
        population.
        Defaults to ``None``.
    population_y_pred_sd : float, optional
        The predicted score standard deviation.
        When the DSM is being calculated for a subgroup,
        this should be the standard deviation for the whole
        population.
        Defaults to None.
    ddof : int, optional
        The delta degrees of freedom. The divisor used in
        calculations is N - ddof, where N represents the
        number of elements.
        Defaults to 1.

    Returns
    -------
    difference_of_std_means : array-like
        The difference of standardized means

    Raises
    ------
    ValueError
        If only one of ``population_y_true_observed_mn`` and
        ``population_y_true_observed_sd`` is not ``None``.
    ValueError
        If only one of ``population_y_pred_mn`` and ``population_y_pred_sd``
        is not ``None``.
    """
    assert len(y_true_observed) == len(y_pred)

    # all of this is just to make sure users aren't passing the population
    # standard deviation and not population mean for either true or predicted
    y_true_observed_population_params = [
        population_y_true_observed_mn,
        population_y_true_observed_sd,
    ]
    y_pred_population_params = [population_y_pred_mn, population_y_pred_sd]

    if len([param for param in y_true_observed_population_params if param is None]) == 1:
        raise ValueError(
            "You must pass both `population_y_true_observed_mn` and "
            "`population_y_true_observed_sd` or neither."
        )

    if len([param for param in y_pred_population_params if param is None]) == 1:
        raise ValueError(
            "You must pass both `population_y_pred_mn` and " "`population_y_pred_sd` or neither."
        )

    warning_msg = (
        "You did not pass population mean and std. for `{}`; "
        "thus, the calculated z-scores will be zero."
    )

    # if the population means and standard deviations were not provided, calculate from the data
    # We only check for mean since the function requires
    # both of these to be set or both to be None
    if population_y_true_observed_mn is None:
        warnings.warn(warning_msg.format("y_true_observed"))
        (population_y_true_observed_sd, population_y_true_observed_mn) = (
            np.std(y_true_observed, ddof=ddof),
            np.mean(y_true_observed),
        )

    if population_y_pred_mn is None:
        warnings.warn(warning_msg.format("y_pred"))
        (population_y_pred_sd, population_y_pred_mn) = (
            np.std(y_pred, ddof=ddof),
            np.mean(y_pred),
        )

    # if any of the standard deviations equal zero
    # raise a warning and return None.
    # We use np.isclose since sometimes sd for float
    # values is a value very close to 0.
    # We use the same tolerance as used for identifying
    # features with zero standard deviation
    if np.isclose(population_y_pred_sd, 0, atol=1e-07) or np.isclose(
        population_y_true_observed_sd, 0, atol=1e-07
    ):
        warnings.warn(
            "Population standard deviations for the computation of "
            "DSM is zero. No value will be computed."
        )
        return None

    # calculate the z-scores for observed and predicted
    y_true_observed_subgroup_z = (
        y_true_observed - population_y_true_observed_mn
    ) / population_y_true_observed_sd
    y_pred_subgroup_z = (y_pred - population_y_pred_mn) / population_y_pred_sd

    # calculate the DSM, given the z-scores for observed and predicted
    difference_of_std_means = np.mean(y_pred_subgroup_z - y_true_observed_subgroup_z)

    return difference_of_std_means


def quadratic_weighted_kappa(y_true_observed, y_pred, ddof=0):  # noqa: D301
    """
    Calculate quadratic-weighted kappa for both discrete and continuous values.

    The formula to compute quadratic-weighted kappa for continuous values
    was developed at ETS by Shelby Haberman.
    See `Haberman (2019) <https://eric.ed.gov/?q=Measures+of+Agreement+Versus
    +Measures+of+Prediction+Accuracy&id=EJ1238497>`_
    for the full derivation. The discrete case is simply treated as a
    special case of the continuous one.

    The formula is as follows:

    :math:`QWK=\\displaystyle\\frac{2*Cov(M,H)}{Var(H)+Var(M)+(\\bar{M}-\\bar{H})^2}`, where

        - :math:`Cov` - covariance with normalization by :math:`N` (total number of observations)
        - :math:`H` - the human score
        - :math:`M` - the system score
        - :math:`\\bar{H}` - mean of :math:`H`
        - :math:`\\bar{M}` - mean of :math:`M`
        - :math:`Var(X)` - variance of X

    Parameters
    ----------
    y_true_observed : array-like
        The observed scores.
    y_pred : array-like
        The predicted scores.
    ddof : int, optional
        Means Delta Degrees of Freedom. The divisor used in
        calculations is N - ddof, where N represents the
        number of elements. When ddof is set to zero, the results
        for discrete case match those from the standard implementations.
        Defaults to 0.

    Returns
    -------
    kappa : float
        The quadratic weighted kappa

    Raises
    ------
    AssertionError
        If the number of elements in ``y_true_observed`` is not equal
        to the number of elements in ``y_pred``.
    """
    assert len(y_true_observed) == len(y_pred)
    y_true_observed_var, y_true_observed_avg = (
        np.var(y_true_observed, ddof=ddof),
        np.mean(y_true_observed),
    )
    y_pred_var, y_pred_avg = (np.var(y_pred, ddof=ddof), np.mean(y_pred))

    numerator = 2 * np.cov(y_true_observed, y_pred, ddof=ddof)[0][1]
    denominator = y_true_observed_var + y_pred_var + (y_true_observed_avg - y_pred_avg) ** 2
    kappa = numerator / denominator
    return kappa
