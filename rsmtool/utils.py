"""
Utility classes and functions.

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

import json
import logging
import re
import warnings

import numpy as np
import pandas as pd

from math import ceil
from glob import glob
from importlib import import_module
from os.path import exists, isabs, join, relpath
from pathlib import Path
from string import Template
from textwrap import wrap
from IPython.display import (display,
                             HTML)

from skll.data import safe_float as string_to_number


HTML_STRING = ("""<li><b>{}</b>: <a href="{}" download>{}</a></li>""")


BUILTIN_MODELS = ['LinearRegression',
                  'EqualWeightsLR',
                  'ScoreWeightedLR',
                  'RebalancedLR',
                  'NNLR',
                  'NNLRIterative',
                  'LassoFixedLambdaThenNNLR',
                  'LassoFixedLambdaThenLR',
                  'PositiveLassoCVThenLR',
                  'LassoFixedLambda',
                  'PositiveLassoCV']

DEFAULTS = {'id_column': 'spkitemid',
            'description': '',
            'description_old': '',
            'description_new': '',
            'train_label_column': 'sc1',
            'test_label_column': 'sc1',
            'human_score_column': 'sc1',
            'exclude_zero_scores': True,
            'use_scaled_predictions': False,
            'use_scaled_predictions_old': False,
            'use_scaled_predictions_new': False,
            'select_transformations': False,
            'standardize_features': True,
            'use_thumbnails': False,
            'use_truncation_thresholds': False,
            'scale_with': None,
            'predict_expected_scores': False,
            'sign': None,
            'features': None,
            'length_column': None,
            'second_human_score_column': None,
            'file_format': 'csv',
            'form_level_scores': None,
            'candidate_column': None,
            'general_sections': ['all'],
            'special_sections': None,
            'custom_sections': None,
            'feature_subset_file': None,
            'feature_subset': None,
            'feature_prefix': None,
            'trim_min': None,
            'trim_max': None,
            'trim_tolerance': 0.4998,
            'subgroups': [],
            'min_n_per_group': None,
            'skll_objective': None,
            'section_order': None,
            'flag_column': None,
            'flag_column_test': None,
            'min_items_per_candidate': None,
            'experiment_names': None}

LIST_FIELDS = ['feature_prefix',
               'general_sections',
               'special_sections',
               'custom_sections',
               'subgroups',
               'section_order',
               'experiment_dirs',
               'experiment_names']

BOOLEAN_FIELDS = ['exclude_zero_scores',
                  'predict_expected_scores',
                  'use_scaled_predictions',
                  'use_scaled_predictions_old',
                  'use_scaled_predictions_new',
                  'use_thumbnails',
                  'use_truncation_thresholds',
                  'select_transformations']

FIELD_NAME_MAPPING = {'expID': 'experiment_id',
                      'LRmodel': 'model',
                      'train': 'train_file',
                      'test': 'test_file',
                      'predictions': 'predictions_file',
                      'feature': 'features',
                      'train.lab': 'train_label_column',
                      'test.lab': 'test_label_column',
                      'trim.min': 'trim_min',
                      'trim.max': 'trim_max',
                      'scale': 'use_scaled_predictions',
                      'feature.subset': 'feature_subset'}

MODEL_NAME_MAPPING = {'empWt': 'LinearRegression',
                      'eqWt': 'EqualWeightsLR',
                      'empWtBalanced': 'RebalancedLR',
                      'empWtDropNeg': '',
                      'empWtNNLS': 'NNLR',
                      'empWtDropNegLasso': 'LassoFixedLambdaThenNNLR',
                      'empWtLasso': 'LassoFixedLambdaThenLR',
                      'empWtLassoBest': 'PositiveLassoCVThenLR',
                      'lassoWtLasso': 'LassoFixedLambda',
                      'lassoWtLassoBest': 'PositiveLassoCV'}

CHECK_FIELDS = {'rsmtool': {'required': ['experiment_id',
                                         'model',
                                         'train_file',
                                         'test_file'],
                            'optional': ['description',
                                         'features',
                                         'feature_subset_file',
                                         'feature_subset',
                                         'file_format',
                                         'sign',
                                         'id_column',
                                         'use_thumbnails',
                                         'train_label_column',
                                         'test_label_column',
                                         'length_column',
                                         'second_human_score_column',
                                         'flag_column',
                                         'flag_column_test',
                                         'exclude_zero_scores',
                                         'trim_min',
                                         'trim_max',
                                         'trim_tolerance',
                                         'predict_expected_scores',
                                         'select_transformations',
                                         'use_scaled_predictions',
                                         'use_truncation_thresholds',
                                         'subgroups',
                                         'min_n_per_group',
                                         'general_sections',
                                         'custom_sections',
                                         'special_sections',
                                         'skll_objective',
                                         'section_order',
                                         'candidate_column',
                                         'standardize_features',
                                         'min_items_per_candidate']},
                'rsmeval': {'required': ['experiment_id',
                                         'predictions_file',
                                         'system_score_column',
                                         'trim_min',
                                         'trim_max'],
                            'optional': ['description',
                                         'id_column',
                                         'human_score_column',
                                         'second_human_score_column',
                                         'file_format',
                                         'flag_column',
                                         'exclude_zero_scores',
                                         'use_thumbnails',
                                         'scale_with',
                                         'trim_tolerance',
                                         'subgroups',
                                         'min_n_per_group',
                                         'general_sections',
                                         'custom_sections',
                                         'special_sections',
                                         'section_order',
                                         'candidate_column',
                                         'min_items_per_candidate']},
                'rsmpredict': {'required': ['experiment_id',
                                            'experiment_dir',
                                            'input_features_file'],
                               'optional': ['id_column',
                                            'candidate_column',
                                            'file_format',
                                            'predict_expected_scores',
                                            'human_score_column',
                                            'second_human_score_column',
                                            'standardize_features',
                                            'subgroups',
                                            'flag_column']},
                'rsmcompare': {'required': ['comparison_id',
                                            'experiment_id_old',
                                            'experiment_dir_old',
                                            'experiment_id_new',
                                            'experiment_dir_new',
                                            'description_old',
                                            'description_new'],
                               'optional': ['use_scaled_predictions_old',
                                            'use_scaled_predictions_new',
                                            'subgroups',
                                            'use_thumbnails',
                                            'general_sections',
                                            'custom_sections',
                                            'special_sections',
                                            'section_order']},
                'rsmsummarize': {'required': ['summary_id',
                                              'experiment_dirs'],
                                 'optional': ['description',
                                              'experiment_names',
                                              'file_format',
                                              'general_sections',
                                              'custom_sections',
                                              'use_thumbnails',
                                              'special_sections',
                                              'subgroups',
                                              'section_order']}}


POSSIBLE_EXTENSIONS = ['csv', 'xlsx', 'tsv']

ID_FIELDS = {'rsmtool': 'experiment_id',
             'rsmeval': 'experiment_id',
             'rsmcompare': 'comparison_id',
             'rsmsummarize': 'summary_id',
             'rsmpredict': 'experiment_id'}

_skll_module = import_module('skll.learner')


def is_skll_model(model_name):
    """
    Check whether the given model is a valid learner name in SKLL.
    Note that the `LinearRegression` model is also available in
    SKLL but we always want to use the built-in model with that name.

    Parameters
    ----------
    model_name : str
        The name of the model to check

    Returns
    -------
    valid: bool
        `True` if the given model name is a valid SKLL learner,
        `False` otherwise
    """
    return hasattr(_skll_module, model_name) and model_name != 'LinearRegression'


def is_built_in_model(model_name):
    """
    Check whether the given model is a valid built-in model.

    Parameters
    ----------
    model_name : str
        The name of the model to check

    Returns
    -------
    valid: bool
        `True` if the given model name is a valid built-in model,
        `False` otherwise
    """
    return model_name in BUILTIN_MODELS


def int_to_float(value):
    """
    Convert integer to float, if possible.

    Parameters
    ----------
    value
        Name of the experiment file we want to locate.

    Returns
    -------
    value
        Value converted to float, if possible
    """
    return float(value) if type(value) == int else value


def convert_to_float(value):
    """
    Convert value to float, if possible.

    Parameters
    ----------
    value
        Name of the experiment file we want to locate.

    Returns
    -------
    value
        Value converted to float, if possible
    """
    return int_to_float(string_to_number(value))


def parse_json_with_comments(filename):
    """
    Parse a JSON file after removing any comments.
    Comments can use either ``//`` for single-line
    comments or or ``/* ... */`` for multi-line comments.

    Parameters
    ----------
    filename : str
        Path to the input JSON file.

    Returns
    -------
    obj : dict
        JSON object representing the input file.

    Note
    ----
    This code was adapted from:
    https://web.archive.org/web/20150520154859/http://www.lifl.fr/~riquetd/parse-a-json-file-with-comments.html
    """

    # Regular expression to identify comments
    comment_re = re.compile(r'(^)?[^\S\n]*/(?:\*(.*?)\*/[^\S\n]*|/[^\n]*)($)?',
                            re.DOTALL | re.MULTILINE)

    with open(filename) as file_buff:
        content = ''.join(file_buff.readlines())

        # Looking for comments
        match = comment_re.search(content)
        while match:

            # single line comment
            content = content[:match.start()] + content[match.end():]
            match = comment_re.search(content)

        # Return JSON object
        config = json.loads(content)
        return config


def compute_expected_scores_from_model(model, featureset, min_score, max_score):
    """
    Compute expected scores using probability distributions over the labels
    from the given SKLL model.

    Parameters
    ----------
    model : skll.Learner
        The SKLL Learner object to use for computing the expected scores.
    featureset : skll.data.FeatureSet
        The SKLL FeatureSet object for which predictions are to be made.
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
        If the given model cannot predict probability distributions and
        or if the score range specified by `min_score` and `max_score`
        does not match what the model predicts in its probability
        distribution.
    """
    if hasattr(model.model, "predict_proba"):
        # Tell the model we want probabiltiies as output. This is likely already set
        # to True but it might not be, e.g., when using rsmpredict.
        model.probability = True
        probability_distributions = model.predict(featureset)
        # check to make sure that the number of labels in the probability
        # distributions matches the number of score points we have
        num_score_points_specified = max_score - min_score + 1
        num_score_points_in_learner = probability_distributions.shape[1]
        if num_score_points_specified != num_score_points_in_learner:
            raise ValueError('The specified number of score points ({}) '
                             'does not match that from the the learner '
                             '({}).'.format(num_score_points_specified,
                                            num_score_points_in_learner))
        expected_scores = probability_distributions.dot(range(min_score, max_score + 1))
    else:
        if model.model_type.__name__ == 'SVC':
            raise ValueError("Expected scores cannot be computed since the SVC model was "
                             "not originally trained to predict probabilities.")
        else:
            raise ValueError("Expected scores cannot be computed since {} is not a "
                             "probabilistic classifier.".format(model.model_type.__name__))

    return expected_scores


def covariance_to_correlation(m):
    """
    This is a port of the R `cov2cor` function.

    Parameters
    ----------
    m : numpy array
        The covariance matrix.

    Returns
    -------
    retval : numpy array
        The cross-correlation matrix.

    Raises
    ------
    ValueError
        If the input matrix is not square.
    """

    # make sure the matrix is square
    numrows, numcols = m.shape
    if not numrows == numcols:
        raise ValueError('Input matrix must be square')

    Is = np.sqrt(1 / np.diag(m))
    retval = Is * m * np.repeat(Is, numrows).reshape(numrows, numrows)
    np.fill_diagonal(retval, 1.0)
    return retval


def partial_correlations(df):
    """
    This is a python port of the `pcor` function implemented in
    the `ppcor` R package, which computes partial correlations
    of each pair of variables in the given data frame `df`,
    excluding all other variables.

    Parameters
    ----------
    df : pd.DataFrame
        Data frame containing the feature values.

    Returns
    -------
    df_pcor : pd.DataFrame
        Data frame containing the partial correlations of of each
        pair of variables in the given data frame `df`,
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
            warnings.warn('When computing partial correlations '
                          'the inverse of the variance-covariance matrix '
                          'was calculated using the Moore-Penrose generalized '
                          'matrix inversion, due to its determinant being at '
                          'or very close to zero.')
        except np.linalg.LinAlgError:
            icvx = empty_array

    pcor = -1 * covariance_to_correlation(icvx)
    np.fill_diagonal(pcor, 1.0)
    df_pcor = pd.DataFrame(pcor, columns=columns, index=columns)
    return df_pcor


def agreement(score1, score2, tolerance=0):
    """
    This function computes the agreement between
    two raters, taking into account the provided
    tolerance.

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

    num_agreements = sum([int(abs(s1 - s2) <= tolerance)
                          for s1, s2 in zip(score1, score2)])

    agreement_value = (float(num_agreements) / len(score1)) * 100
    return agreement_value


def standardized_mean_difference(y_true_observed,
                                 y_pred,
                                 population_y_true_observed_sd=None,
                                 population_y_pred_sd=None,
                                 method='unpooled',
                                 ddof=1):
    """
    This function computes the standardized mean
    difference between a system score and human score.
    The numerator is calculated as mean(y_pred) - mean(y_true_observed)
    for all of the available methods.

    Parameters
    ----------
    y_true_observed : array-like
        The observed scores for the group or subgroup.
    y_pred : array-like
        The predicted score for the group or subgroup.
    population_y_true_observed_sd : float or None, optional
        The population true score standard deviation.
        When the SMD is being calculated for a subgroup,
        this should be the standard deviation for the whole
        population.
        Defaults to None.
    population_y_pred_sd : float or None, optional
        The predicted score standard deviation.
        When the SMD is being calculated for a subgroup,
        this should be the standard deviation for the whole
        population.
        Defaults to None.
    method : {'williamson', 'johnson', 'pooled', 'unpooled'}, optional
        The SMD method to use.
        - `williamson`: Denominator is the pooled population standard deviation of `y_true_observed` and `y_pred`.
        - `johnson`: Denominator is the population standard deviation of `y_true_observed`.
        - `pooled`: Denominator is the pooled standard deviation of `y_true_observed` and `y_pred for this group.
        - `unpooled`: Denominator is the standard deviation of `y_true_observed` for this group.

        Defaults to 'unpooled'.
    ddof : int, optional
        Means Delta Degrees of Freedom. The divisor used in
        calculations is N - ddof, where N represents the
        number of elements. By default ddof is zero.
        Defaults to 1.

    Returns
    -------
    smd : float
        The SMD for the given group or subgroup.

    Raises
    ------
    ValueError
        If method='williamson' and either population_y_true_observed_sd or
        population_y_pred_sd is None.
    ValueError
        If method is not in {'unpooled', 'pooled', 'williamson', 'johnson'}.

    Notes
    -----
    The 'williamson' implementation was recommended by Williamson, et al. (2012).
    The metric is only applicable when both sets of scores are on the same
    scale.
    """
    numerator = np.mean(y_pred) - np.mean(y_true_observed)

    method = method.lower()
    if method == 'unpooled':
        denominator = np.std(y_true_observed, ddof=ddof)
    elif method == 'pooled':
        denominator = np.sqrt((np.std(y_true_observed, ddof=ddof)**2 +
                               np.std(y_pred, ddof=ddof)**2) / 2)
    elif method == 'johnson':
        if population_y_true_observed_sd is None:
            raise ValueError("If `method='johnson'`, then `population_y_true_observed_sd` "
                             "must be provided.")
        denominator = population_y_true_observed_sd
    elif method == 'williamson':
        if population_y_true_observed_sd is None or population_y_pred_sd is None:
            raise ValueError("If `method='williamson'`, both `population_y_true_observed_sd` "
                             "and `population_y_pred_sd` must be provided.")
        denominator = np.sqrt((population_y_true_observed_sd**2 +
                               population_y_pred_sd**2) / 2)
    else:
        possible_methods = {"'unpooled'", "'pooled'", "'johnson'", "'williamson'"}
        raise ValueError("The available methods are {{{}}}; you selected {}."
                         "".format(', '.join(possible_methods), method))

    # if the denominator is zero, then return NaN as the SMD
    smd = np.nan if denominator == 0 else numerator / denominator
    return smd


def difference_of_standardized_means(y_true_observed,
                                     y_pred,
                                     population_y_true_observed_mn=None,
                                     population_y_pred_mn=None,
                                     population_y_true_observed_sd=None,
                                     population_y_pred_sd=None,
                                     ddof=1):
    """
    Calculate the difference of standardized means. This is done by standardizing
    both observed and predicted scores to z-scores using mean and standard deviation
    for the whole population and then calculating differences between standardized
    means for each subgroup.

    Parameters
    ----------
    y_true_observed : array-like
        The observed scores for the group or subgroup.
    y_pred : array-like
        The predicted score for the group or subgroup.
        The predicted scores.
    population_y_true_observed_mn : float or None, optional
        The population true score mean.
        When the DSM is being calculated for a subgroup,
        this should be the mean for the whole
        population.
        Defaults to None.
    population_y_pred_mn : float or None, optional
        The predicted score mean.
        When the DSM is being calculated for a subgroup,
        this should be the mean for the whole
        population.
        Defaults to None.
    population_y_true_observed_sd : float or None, optional
        The population true score standard deviation.
        When the DSM is being calculated for a subgroup,
        this should be the standard deviation for the whole
        population.
        Defaults to None.
    population_y_pred_sd : float or None, optional
        The predicted score standard deviation.
        When the DSM is being calculated for a subgroup,
        this should be the standard deviation for the whole
        population.
        Defaults to None.
    ddof : int, optional
        Means Delta Degrees of Freedom. The divisor used in
        calculations is N - ddof, where N represents the
        number of elements. By default ddof is zero.
        Defaults to 1.

    Returns
    -------
    difference_of_std_means : array-like
        The difference of standardized means

    Raises
    ------
    ValueError
        If only one of `population_y_true_observed_mn` and `population_y_true_observed_sd` is
        not None.
    ValueError
        If only one of `population_y_pred_mn` and `population_y_pred_sd` is
        not None.
    """
    assert len(y_true_observed) == len(y_pred)

    # all of this is just to make sure users aren't passing the population
    # standard deviation and not population mean for either true or predicted
    y_true_observed_population_params = [population_y_true_observed_mn,
                                         population_y_true_observed_sd]
    y_pred_population_params = [population_y_pred_mn,
                                population_y_pred_sd]

    if any(y_true_observed_population_params) and not all(y_true_observed_population_params):
        raise ValueError('You must pass both `population_y_true_observed_mn` and '
                         '`population_y_true_observed_sd` or neither.')

    if any(y_pred_population_params) and not all(y_pred_population_params):
        raise ValueError('You must pass both `population_y_pred_mn` and '
                         '`population_y_pred_sd` or neither.')

    warning_msg = ('You did not pass population mean and std. for `{}`; '
                   'thus, the calculated z-scores will be zero.')

    # if the population means and standard deviations were not provided, calculate from the data
    if not population_y_true_observed_mn or not population_y_true_observed_sd:

        warnings.warn(warning_msg.format('y_true_observed'))
        (population_y_true_observed_sd,
         population_y_true_observed_mn) = (np.std(y_true_observed, ddof=ddof),
                                           np.mean(y_true_observed))

    if not population_y_pred_mn or not population_y_pred_sd:

        warnings.warn(warning_msg.format('y_pred'))
        (population_y_pred_sd,
         population_y_pred_mn) = (np.std(y_pred, ddof=ddof),
                                  np.mean(y_pred))

    # calculate the z-scores for observed and predicted
    y_true_observed_subgroup_z = ((y_true_observed - population_y_true_observed_mn) /
                                  population_y_true_observed_sd)
    y_pred_subgroup_z = ((y_pred - population_y_pred_mn) /
                         population_y_pred_sd)

    # calculate the DSM, given the z-scores for observed and predicted
    difference_of_std_means = np.mean(y_pred_subgroup_z - y_true_observed_subgroup_z)

    return difference_of_std_means


def quadratic_weighted_kappa(y_true_observed, y_pred, ddof=0):
    """
    Calculate Quadratic-weighted Kappa that works
    for both discrete and continuous values.

    The formula to compute quadratic-weighted kappa
    for continuous values was developed at ETS by
    Shelby Haberman. See `Haberman (2019) <https://onlinelibrary.wiley.com/doi/abs/10.1002/ets2.12258>`_. for the full
    derivation. The discrete case is simply treated as
    a special case of the continuous one.

    The formula is as follows:

    :math:`QWK=\\displaystyle\\frac{2*Cov(M,H)}{Var(H)+Var(M)+(\\bar{M}-\\bar{H})^2}`, where

        - :math:`Cov` - covariance with normalization by :math:`N` (the total number of observations given)
        - :math:`H` - the human score
        - :math:`M` - the system score
        - :math:`\\bar{H}` - mean of :math:`H`
        - :math:`\\bar{M}` - the mean of :math:`M`
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
    y_true_observed_var, y_true_observed_avg = (np.var(y_true_observed, ddof=ddof),
                                                np.mean(y_true_observed))
    y_pred_var, y_pred_avg = (np.var(y_pred, ddof=ddof),
                              np.mean(y_pred))

    numerator = 2 * np.cov(y_true_observed, y_pred, ddof=ddof)[0][1]
    denominator = y_true_observed_var + y_pred_var + (y_true_observed_avg - y_pred_avg)**2
    kappa = numerator / denominator
    return kappa


def float_format_func(num, prec=3):
    """
    Format the given floating point number to the specified precision
    and return as a string.

    Parameters:
    ----------
    num : float
        The floating point number to format.
    prec: int, optional
        The number of decimal places to use when displaying the number.
        Defaults to 3.

    Returns:
    -------
    ans: str
        The formatted string representing the given number.
    """

    formatter_string = Template('{:.${prec}f}').substitute(prec=prec)
    ans = formatter_string.format(num)
    return ans


def int_or_float_format_func(num, prec=3):
    """
    Identify whether the number is float or integer. When displaying
    integers, use no decimal. For a float, round to the specified
    number of decimal places. Return as a string.

    Parameters:
    -----------
    num : float or int
        The number to format and display.
    prec : int, optional
        The number of decimal places to display if x is a float.
        Defaults to 3.

    Returns:
    -------
    ans : str
        The formatted string representing the given number.
    """

    if float.is_integer(num):
        ans = '{}'.format(int(num))
    else:
        ans = float_format_func(num, prec=prec)
    return ans


def custom_highlighter(num,
                       low=0,
                       high=1,
                       prec=3,
                       absolute=False,
                       span_class='bold'):
    """
    Return the supplied float as an HTML <span> element with the specified
    class if its value is below ``low`` or above ``high``. If its value does
    not meet those constraints, then return as a plain string with the
    specified number of decimal places.

    Parameters:
    -----------
    num : float
        The floating point number to format.
    low : float
        The number will be displayed as an HTML span it is below this value.
        Defaults to 0.
    high : float
        The number will be displayed as an HTML span it is above this value.
        Defaults to 1.
    prec : int
        The number of decimal places to display for x. Defaults to 3.
    absolute: bool
        If True, use the absolute value of x for comparison.
        Defaults to False.
    span_class: str
        One of ``bold`` or ``color``. These are the two classes
        available for the HTML span tag.

    Returns:
    --------
    ans : str
        The formatted (plain or HTML) string representing the given number.
    """
    abs_num = abs(num) if absolute else num
    val = float_format_func(num, prec=prec)
    ans = ('<span class="highlight_{}">{}</span>'.format(span_class, val)
           if abs_num < low or abs_num > high else val)
    return ans


def bold_highlighter(num, low=0, high=1, prec=3, absolute=False):
    """
    Instantiating ``custom_highlighter()`` with the ``bold`` class as
    the default.

    Parameters:
    -----------
    num : float
        The floating point number to format.
    low : float
        The number will be displayed as an HTML span it is below this value.
        Defaults to 0.
    high : float
        The number will be displayed as an HTML span it is above this value.
        Defaults to 1.
    prec : int
        The number of decimal places to display for x.
        Defaults to 3.
    absolute: bool
        If True, use the absolute value of x for comparison.
        Defaults to False.

    Returns:
    --------
    ans : str
        The formatted highlighter with bold class as default.
    """
    ans = custom_highlighter(num, low, high, prec, absolute, 'bold')
    return ans


def color_highlighter(num, low=0, high=1, prec=3, absolute=False):
    """
    Instantiating ``custom_highlighter()`` with the ``color`` class as
    the default.

    Parameters:
    -----------
    num : float
        The floating point number to format.
    low : float
        The number will be displayed as an HTML span it is below this value.
        Defaults to 0.
    high : float
        The number will be displayed as an HTML span it is above this value.
        Defaults to 1.
    prec : int
        The number of decimal places to display for x.
        Defaults to 3.
    absolute: bool
        If True, use the absolute value of x for comparison.
        Defaults to False.

    Returns:
    --------
    ans : str
        The formatted highlighter with color class as default.
    """
    ans = custom_highlighter(num, low, high, prec, absolute, 'color')
    return ans


def compute_subgroup_plot_params(group_names,
                                 num_plots):
    """
    Computing subgroup plot and figure parameters based on number of
    subgroups and number of plots to be generated.

    Parameters
    ----------
    group_names : list
        A list of subgroup names for plots.
    num_plots : int
        The number of plots to compute.

    Returns
    -------
    figure_width : int
        The width of the figure.
    figure_height : int
        The height of the figure.
    num_rows : int
        The number of rows for the plots.
    num_columns : int
        The number of columns for the plots.
    wrapped_group_names : list of str
        A list of group names for plots.
    """
    wrapped_group_names = ['\n'.join(wrap(str(gn), 20)) for gn in group_names]
    plot_height = 4 if wrapped_group_names == group_names else 6
    num_groups = len(group_names)
    if num_groups <= 6:
        num_columns = 2
        num_rows = ceil(num_plots / num_columns)
        figure_width = num_columns * num_groups
        figure_height = plot_height * num_rows
    else:
        num_columns = 1
        num_rows = num_plots
        figure_width = 10
        figure_height = plot_height * num_plots

    return (figure_width, figure_height, num_rows, num_columns, wrapped_group_names)


def has_files_with_extension(directory, ext):
    """
    Check if the directory has any files with the given extension.

    Parameters
    ----------
    directory : str
        The path to the directory where output is located.
    ext : str
        The the given extension.

    Returns
    -------
    bool
        True if directory contains files with given extension,
        else False.
    """
    files_with_extension = glob(join(directory, '*.{}'.format(ext)))
    return len(files_with_extension) > 0


def get_output_directory_extension(directory, experiment_id):
    """
    Check the output directory to determine what file extensions
    exist. If more than one extension (in the possible list of
    extensions) exists, then raise a ValueError. Otherwise,
    return the one file extension. If no extensions can be found, then
    `csv` will be returned by default.

    Possible extensions include: `csv`, `tsv`, `xlsx`. Files in the
    directory with none of these extensions will be ignored.

    Parameters
    ----------
    directory : str
        The path to the directory where output is located.
    experiment_id : str
        The ID of the experiment.

    Returns
    -------
    extension : {'csv', 'tsv', 'xlsx'}
        The extension that output files in this directory
        end with.

    Raises
    ------
    ValueError
        If any files in the directory have different extensions,
        and are in the list of possible output extensions.
    """
    extension = 'csv'
    extensions_identified = {ext for ext in POSSIBLE_EXTENSIONS
                             if has_files_with_extension(directory, ext)}

    if len(extensions_identified) > 1:
        raise ValueError('Some of the files in the experiment output directory (`{}`) '
                         'for `{}` have different extensions. All files in this directory '
                         'must have the same extension. The following extensions were '
                         'identified : {}'.format(directory,
                                                  experiment_id,
                                                  ', '.join(extensions_identified)))

    elif len(extensions_identified) == 1:
        extension = list(extensions_identified)[0]

    return extension


def get_thumbnail_as_html(path_to_image, image_id, path_to_thumbnail=None):
    """
    Given an path to an image file, generate the HTML for
    a click-able thumbnail version of the image.
    On click, this HTML will open the full-sized version
    of the image in a new window.

    Parameters
    ----------
    path_to_image : str
        The absolute or relative path to the image.
        If an absolute path is provided, it will be
        converted to a relative path.
    image_id : int
        The id of the <img> tag in the HTML. This must
        be unique for each <img> tag.
    path_to_thumbnail : str or None, optional
        If you would like to use a different thumbnail
        image, specify the path to this thumbnail.
        Defaults to None.

    Returns
    -------
    image : str
        The HTML string generated for the image.

    Raises
    ------
    FileNotFoundError
        If the image file cannot be located.
    """
    error_message = 'The file `{}` could not be located.'
    if not exists(path_to_image):
        raise FileNotFoundError(error_message.format(path_to_image))

    # check if the path is relative or absolute
    if isabs(path_to_image):
        rel_image_path = relpath(path_to_image)
    else:
        rel_image_path = path_to_image

    # if `path_to_thumbnail` is None, use `path_to_image`;
    # otherwise, get the relative path to the thumbnail
    if path_to_thumbnail is None:
        rel_thumbnail_path = rel_image_path
    else:
        if not exists(path_to_thumbnail):
            raise FileNotFoundError(error_message.format(path_to_thumbnail))
        rel_thumbnail_path = relpath(path_to_thumbnail)

    # specify the thumbnail style
    style = """
    <style>
    img {
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 5px;
        width: 150px;
        cursor: pointer;
    }
    </style>
    """

    # on click, open larger image in new window
    script = """
    <script>
    function getPicture(picpath) {{
        window.open(picpath, 'Image', resizable=1);
    }};
    </script>""".format(image_id)

    # generate image tags
    image = ("""<img id='{}' src='{}' onclick='getPicture("{}")' """
             """title="Click to enlarge">"""
             """</img>""").format(image_id,
                                  rel_image_path,
                                  rel_thumbnail_path)

    # create the image HTML
    image += style
    image += script
    return image


def show_thumbnail(path_to_image, image_id, path_to_thumbnail=None):
    """
    Given an path to an image file, display
    a click-able thumbnail version of the image.
    On click, open the full-sized version of the
    image in a new window.

    Parameters
    ----------
    path_to_image : str
        The absolute or relative path to the image.
        If an absolute path is provided, it will be
        converted to a relative path.
    image_id : int
        The id of the <img> tag in the HTML. This must
        be unique for each <img> tag.
    path_to_thumbnail : str or None, optional
        If you would like to use a different thumbnail
        image, specify the path to the thumbnail.
        Defaults to None.

    Displays
    --------
    display : IPython.core.display.HTML
        The HTML display of the thumbnail image.
    """
    display(HTML(get_thumbnail_as_html(path_to_image,
                                       image_id,
                                       path_to_thumbnail)))


def get_files_as_html(output_dir, experiment_id, file_format, replace_dict={}):
    """
    Generate HTML list items for each file name,
    given output directory. Optionally pass a
    replacement dictionary to use more descriptive
    titles for the file names.

    Parameters
    ----------
    output_dir : str
        The output directory.
    experiment_id : str
        The experiment ID.
    file_format : str
        The format of the output files.
    replace_dict : dict, optional
        A dictionary which makes file names to descriptions.
        Defaults to empty dictionary.

    Returns
    ------
    html_string : str
        HTML string with file descriptions and links.
    """
    output_dir = Path(output_dir)
    parent_dir = output_dir.parent
    files = output_dir.glob('*.{}'.format(file_format))
    html_string = ''
    for file in sorted(files):
        relative_file = ".." / file.relative_to(parent_dir)
        relative_name = relative_file.stem.replace('{}_'.format(experiment_id), '')

        # check if relative name is in the replacement dictionary and,
        # if it is, use the more descriptive name in the replacement
        # dictionary. Otherwise, normalize the file name and use that
        # as the description instead.
        if relative_name in replace_dict:
            descriptive_name = replace_dict[relative_name]
        else:
            descriptive_name_components = relative_name.split('_')
            descriptive_name = ' '.join(descriptive_name_components).title()

        html_string += HTML_STRING.format(descriptive_name,
                                          relative_file,
                                          file_format)

    return """<ul><html>""" + html_string + """</ul></html>"""


def show_files(output_dir, experiment_id, file_format, replace_dict={}):
    """
    Show files for a given output directory.

    Parameters
    ----------
    output_dir : str
        The output directory.
    experiment_id : str
        The experiment ID.
    file_format : str
        The format of the output files.
    replace_dict : dict, optional
        A dictionary which makes file names to descriptions.
        Defaults to empty dictionary.

    Displays
    --------
    display : IPython.core.display.HTML
        The HTML file descriptions and links.
    """
    html_string = get_files_as_html(output_dir,
                                    experiment_id,
                                    file_format,
                                    replace_dict)
    display(HTML(html_string))


class LogFormatter(logging.Formatter):
    """
    Custom logging formatter.

    Adapted from:
        http://stackoverflow.com/questions/1343227/
        can-pythons-logging-format-be-modified-depending-
        on-the-message-log-level
    """

    info_fmt = "%(msg)s"
    warn_fmt = "WARNING: %(msg)s"

    err_fmt = "ERROR: %(msg)s"
    dbg_fmt = "DEBUG: %(module)s: %(lineno)d: %(msg)s"

    def __init__(self, fmt="%(levelno)s: %(msg)s"):

        logging.Formatter.__init__(self, fmt)

    def format(self, record):
        """
        format the logger

        Parameters
        ----------
        record
            The record to format
        """

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._fmt

        # Replace the original format with one customized by logging level
        if record.levelno == logging.DEBUG:
            self._fmt = LogFormatter.dbg_fmt
            self._style = logging.PercentStyle(self._fmt)

        elif record.levelno == logging.WARNING:
            self._fmt = LogFormatter.warn_fmt
            self._style = logging.PercentStyle(self._fmt)

        elif record.levelno == logging.INFO:
            self._fmt = LogFormatter.info_fmt
            self._style = logging.PercentStyle(self._fmt)

        elif record.levelno == logging.ERROR:
            self._fmt = LogFormatter.err_fmt
            self._style = logging.PercentStyle(self._fmt)

        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._fmt = format_orig

        return result
