import json
import logging
import os

import numpy as np
import pandas as pd

from os import makedirs
from os.path import join
from string import Template


# get the path to this file
package_path = os.path.dirname(__file__)


class LogFormatter(logging.Formatter):
    """
    Custom logging formatter.

    Adapted from: http://stackoverflow.com/questions/1343227/can-pythons-logging-format-be-modified-depending-on-the-message-log-level
    """

    err_fmt = "ERROR: %(msg)s"
    warn_fmt = "WARNING: %(msg)s"
    dbg_fmt = "DBG: %(module)s: %(lineno)d: %(msg)s"
    info_fmt = "%(msg)s"

    def __init__(self, fmt="%(levelno)s: %(msg)s"):
        logging.Formatter.__init__(self, fmt)

    def format(self, record):

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
    df : pandas DataFrame
        Data frame containing the feature values.

    Returns
    -------
    df_pcor : pandas DataFrame
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
        # we also return nans if there is singularity in the data (e.g. all human scores are the same)
        try:
            icvx = np.linalg.inv(df_cov)
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


def write_experiment_output(data_frames,
                            suffixes,
                            experiment_id,
                            csvdir,
                            reset_index=False):
    """
    Write out each of the given list of data frames as a ``.csv`` file
    in the given directory. Each data frame was generated as part of
    running an RSMTool exepriment. All ``.csv`` files are prefixed with
    the given experiment ID and suffixed with the corresponding value in
    the list of suffixes. Additionally, the indexes in the data frames
    are reset if so specified.

    Parameters
    ----------
    data_frames : list of pandas DataFrame
        List of data frames to write out.
    suffixes : list of str
        List of suffixes, one for each of the data frames.
    experiment_id : str
        The experiment ID.
    csvdir : str
        Path to the `output` experiment sub-directory that will
        contain the CSV files corresponding to each of the data frames.
    reset_index : bool, optional
        Whether to reset the index of each data frame
        before writing to disk. Defaults to `False`.
    """
    for df, suffix in zip(data_frames, suffixes):

        # if the data frame is empty, skip it
        if df.empty:
            continue

        # reset the index if we are asked to
        if reset_index:
            df.index.name = ''
            df.reset_index(inplace=True)

        # write data frame to CSV file
        outfile = join(csvdir, '{}_{}.csv'.format(experiment_id, suffix))
        df.to_csv(outfile, index=False)


def write_feature_json(feature_specs,
                       selected_features,
                       experiment_id,
                       featuredir):
    """
    Write out the feature ``.json`` file to disk.

    Parameters
    ----------
    feature_specs : dict
        Dictionary containing the specifications of the features.
    selected_features : list of str
        List of features that were selected for model building.
    experiment_id : str
        The experiment ID.
    featuredir : str
        Path to the `feature` experiment output directory where the
        feature JSON file will be saved.
    """
    feature_specs_selected = {}
    feature_specs_selected['features'] = [feature_info for feature_info in feature_specs['features'] if feature_info['feature'] in selected_features]

    makedirs(featuredir, exist_ok=True)
    outjson = join(featuredir, '{}_selected.json'.format(experiment_id))
    with open(outjson, 'w') as outfile:
        json.dump(feature_specs_selected, outfile, indent=4, separators=(',', ': '))


def scale_coefficients(intercept,
                       coefficients,
                       feature_names,
                       train_predictions_mean,
                       train_predictions_sd,
                       h1_sd):
    """
    Scale coefficients and intercept using human scores and model
    prediction on the training set. This procedure approximates
    what is done in operational setting but does not apply
    trimming to predictions.

    Parameters
    ----------
    intercept : float
        The model intercept value.
    coefficients : numpy array
        Numpy array containing the model coefficients.
    feature_names : list of str
        List of feature names corresponding to the coefficients.
    train_predictions_mean : float
        The mean of the predictions on the training set.
    train_predictions_sd : float
        The std. dev. of the predictions on the training set.
    h1_sd : float
        The std. dev. of the H1 scores.

    Returns
    -------
    df_scaled_coefficients : pandas DataFrame
        Data frame containing the scaled coefficients
        and the feature names, along with the intercept.
    """

    # scale the coefficients and the intercept
    scaled_coefficients = coefficients * h1_sd/train_predictions_sd

    # adjust the intercept to set the mean predicted score
    # to the mean of the training variable
    new_intercept = intercept * (h1_sd/train_predictions_sd) + train_predictions_mean * (1 - h1_sd/train_predictions_sd)

    intercept_and_feature_names = ['Intercept'] + feature_names
    intercept_and_feature_values = [new_intercept] + list(scaled_coefficients)

    # create a data frame with new values
    df_scaled_coefficients = pd.DataFrame({'feature': intercept_and_feature_names,
                                           'coefficient': intercept_and_feature_values},
                                          columns=['feature', 'coefficient'])

    return df_scaled_coefficients


def float_format_func(num, prec=3):
    """
    Format the given floating point number to the specified precision
    and return as a string.

    Parameters:
    ----------
    num : float
        The floating point number to format.

    prec: int
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
    prec : int
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
    ans = '<span class="highlight_{}">{}</span>'.format(span_class, val) if abs_num < low or abs_num > high else val
    return ans


def bold_highlighter(num, low=0, high=1, prec=3, absolute=False):
    """
    Instantiating ``custom_highlighter()`` with the ``bold`` class as
    the default.
    """
    ans = custom_highlighter(num, low, high, prec, absolute, 'bold')
    return ans


def color_highlighter(num, low=0, high=1, prec=3, absolute=False):
    """
    Instantiating ``custom_highlighter()`` with the ``color`` class as
    the default.
    """
    ans = custom_highlighter(num, low, high, prec, absolute, 'color')
    return ans
