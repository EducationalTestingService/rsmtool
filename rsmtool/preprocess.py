"""
Functions dealing with preprocessing input data.

:author: Nitin Madnani (nmadnani@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:organization: ETS
"""

import logging

import numpy as np
import pandas as pd


def trim(values, trim_min, trim_max):
    """
    Trim the values contained in the given numpy array to `trim_min` - 0.49998 as the floor and `trim_max` + 0.49998
    as the ceiling.

    Parameters
    ----------
    values : list of float
        List of values to trim.
    trim_min : float
        The lowest score on the score point, used for
        trimming the raw regression predictions.
    trim_max : float
        The highest score on the score point, used for
        trimming the raw regression predictions.

    Returns
    -------
    trimmed_values : list of float
        List of trimmed values.
    """

    new_max = trim_max + 0.49998
    new_min = trim_min - 0.49998
    trimmed_values = values.copy()
    trimmed_values[trimmed_values > new_max] = new_max
    trimmed_values[trimmed_values < new_min] = new_min
    return trimmed_values


def filter_on_flag_columns(df, flag_column_dict):
    """
    Check that all flag_columns are present in the given
    data frame, convert these columns to strings and filter
    out the values which do not match the condition in
    `flag_column_dict`.

    Parameters
    ----------
    df : pandas DataFrame
        Data frame containing the feature values.
    flag_column_dict : dict
        Dictionary containing the flag column
        information.

    Returns
    -------
    df_responses_with_requested_flags : pandas DataFrame
        Data frame containing the responses remaining
        after filtering using the specified flag
        columns.
    df_responses_with_excluded_flags : pandas DataFrame
        Data frame containing the responses filtered
        out using the specified flag columns.

    Raises
    ------
    KeyError
        If the columns listed in the dictionary are
        not actually present in the data frame.
    ValueError
        If no responses remain after filtering based
        on the flag column information.
    """

    df_new = df.copy()

    flag_columns = list(flag_column_dict.keys())

    if not flag_columns:
        return df_new, pd.DataFrame(columns=df.columns)
    else:
        # check that all columns are present
        missing_flag_columns = set(flag_columns).difference(df_new.columns)
        if missing_flag_columns:
            raise KeyError("The data does not contain columns "
                           "for all flag columns specified in the "
                           "configuration file. Please check for "
                           "capitalization and other spelling "
                           "errors and make sure the flag column "
                           "names do not contain hyphens. "
                           "The data does not have the following columns: "
                           "{}".format(', '.join(missing_flag_columns)))

        # since flag column may be a mix of strings and numeric values
        # we convert all integers to floats so that 1 and 1.0
        # are treated as the same value

        convert_to_float = lambda x: float(x) if type(x) == int else x

        # we first convert the values in the dictionary
        flag_column_dict_to_float = dict([(key, list(map(convert_to_float, value)))
                                         for (key, value) in flag_column_dict.items()])

        # and then the values in the data
        for column in flag_columns:
            df_new[column] = df_new[column].map(convert_to_float)

        # identify responses with values which satisfy the condition
        full_mask = df_new.isin(flag_column_dict_to_float)
        flagged_mask = full_mask[list(flag_column_dict_to_float.keys())].all(1)
        df_responses_with_requested_flags = df_new[flagged_mask]
        df_responses_with_excluded_flags = df_new[~flagged_mask]

        # make sure that the remaining data frame is not empty
        if len(df_responses_with_requested_flags) == 0:
            raise ValueError("No responses remaining after filtering "
                             "on flag columns. No further analysis can "
                             "be run.")

        return (df_responses_with_requested_flags,
                df_responses_with_excluded_flags)


def filter_on_column(df,
                     column,
                     id_column,
                     exclude_zeros=False,
                     exclude_zero_sd=False):
    """
    Flter out the rows in the given data frame that contain non-numeric
    (or zero, if specified) values in the specified column. Additionally,
    it may exclude any columns if they have a standard deviation
    (:math:`\\sigma`) of 0.

    Parameters
    ----------
    df : pandas DataFrame
        Input data frame containing the feature values.
    column : str
        Name of the column from which to filter out values.
    id_column : str
        Name of the column containing the unique response IDs.
    exclude_zeros : bool, optional
        Whether to exclude responses containing zeros
        in the specified column. Defaults to `False`.
    exclude_zero_sd : bool, optional
        Whether to perform the additional filtering step of removing
        columns that have :math:`\\sigma = 0`. Defaults to `False`.

    Returns
    -------
    df_filtered : pandas DataFrame
        Data frame containing the responses that were *not* filtered out.
    df_excluded : pandas DataFrame
        Data frame containing the non-numeric or zero responses that
        were filtered out.

    Note
    ----
    The columns with :math:`\\sigma=0` are removed from both output
    data frames.
    """

    logger = logging.getLogger(__name__)

    # create a copy of the original data frame
    df_filtered = df.copy()

    # return a copy of the original data frame if
    # the given column does not exist at all
    if not column in df.columns:
        return df_filtered

    # Force convert the label column to numeric and
    # convert whatever can't be converted to a NaN
    df_filtered[column] = pd.to_numeric(df_filtered[column], errors='coerce').astype(float)

    # Save the values that have been converted to NaNs
    # as a separate data frame. We want to keep them as NaNs
    # to do more analyses later.
    bad_rows = df_filtered[df_filtered[column].isnull()]

    # drop the NaNs that we might have gotten
    df_filtered = df_filtered[df_filtered[column].notnull()]

    # exclude zeros if specified
    if exclude_zeros:
        zero_indices = df_filtered[df_filtered[column] == 0].index.values
        zero_rows = df.loc[zero_indices]
        df_filtered = df_filtered[df_filtered[column] != 0]
    else:
        zero_rows = pd.DataFrame()

    # combine all the filtered rows into a single data frame
    df_excluded = pd.concat([bad_rows, zero_rows])

    # reset the index so that the indexing works correctly
    # for the next feature with missing values
    df_filtered.reset_index(drop=True, inplace=True)
    df_excluded.reset_index(drop=True, inplace=True)

    # Drop this column if the standard deviation equals zero:
    # for training set sd == 0 will break normalization.
    # We set the tolerance level to the 6th digit
    # to account for a possibility that the exact value
    # computed by std is not 0
    if exclude_zero_sd is True:
        feature_sd = df_filtered[column].std()
        if np.isclose(feature_sd, 0, atol=1e-06):
            logger.info("Feature {} was excluded from the model"
                        " because its standard deviation in the "
                        "training set is equal to 0.".format(column))
            df_filtered = df_filtered.drop(column, 1)
            df_excluded = df_excluded.drop(column, 1)

    # return the filtered rows and the new data frame
    return (df_filtered, df_excluded)


def remove_outliers(values, mean=None, sd=None, sd_multiplier=4):
    """
    Clamp any values in the given numpy array that are
    +/- `sd_multiplier` (:math:`m`) standard deviations (:math:`\\sigma`) away
    from the mean (:math:`\\mu`). Use given `mean` and `sd` instead
    of computing :math:`\\sigma` and :math:`\\mu`, if specified.

    The values are clamped to the interval:

    .. math::

        [\\mu - m * \\sigma, \\mu + m * \\sigma]

    Parameters
    ----------
    values : numpy array
        Numpy array containing values for a feature.
    mean : None, optional
        Use the given mean value when computing outliers
        instead of the mean from the data.
    sd : None, optional
        Use the given std. dev. value when computing
        outliers instead of the std. dev. from the
        data.
    sd_multiplier : int, optional
        Use the given multipler for the std. dev. when
        computing the outliers. Defaults to 4.

    Returns
    -------
    new_values : numpy array
        Numpy array with the outliers clamped.
    """

    # convert data to a numpy float array before doing any clamping
    new_values = np.array(values, dtype=np.float)

    if not mean:
        mean = new_values.mean()
    if not sd:
        sd = new_values.std()

    floor = mean - sd_multiplier * sd
    ceiling = mean + sd_multiplier * sd

    new_values[new_values > ceiling] = ceiling
    new_values[new_values < floor] = floor

    return new_values


def apply_inverse_transform(name, data,
                            raise_error=True,
                            sd_multiplier=4,):
    """
    Apply the inverse transform to `data`.

    Parameters
    ----------
    name : str
        Name of the feature to transform.
    data : numpy array
        Numpy array containing the feature values.
    raise_error : bool, optional
        When set to true, raises an error if the transform is applied to
         a feature that can be zero or to a feature that can have different signs.
    sd_multiplier : int, optional
        Use this std. dev. multiplier to compute the ceiling
        and floor for outlier removal and check that these
        are not equal to zero.


    Returns
    -------
    new_data: numpy array
        Numpy array containing the transformed feature
        values.

    Raises
    ------
    ValueError
        If the transform is applied to a feature that can
        be zero or to a feature that can have different
        signs and `raise_error` is set to 'True'
    """

    if np.any(data == 0):
        if raise_error:
            raise ValueError("The inverse transformation should not be "
                             "applied to feature {} which can have a "
                             "value of 0".format(name))
        else:
            logging.warning("The inverse transformation was applied "
                            "feature {} which has a value of 0 for "
                            "some responses. No system score will be "
                            "generated for such responses".format(name))

    # check if the floor or ceiling are zero
    data_mean = np.mean(data)
    data_sd = np.std(data, ddof=1)
    floor = data_mean - sd_multiplier * data_sd
    ceiling = data_mean + sd_multiplier * data_sd
    if floor == 0 or ceiling == 0:
        logging.warning("The floor/ceiling for feature {} is zero "
                        "after applying the inverse transformation".format(name))

    # check if the feature can be both positive and negative
    all_positive = np.all(np.abs(data) == data)
    all_negative = np.all(np.abs(data) == -data)
    if not (all_positive or all_negative):
        if raise_error:
            raise ValueError("The inverse transformation should not be "
                             "applied to feature {} where the values can "
                             "have different signs".format(name))
        else:
            logging.warning("The inverse transformation was "
                            "applied to feature {} where the values can"
                            "have different signs. This can change "
                            "the ranking of the responses".format(name))

    new_data = 1 / data
    return new_data


def apply_sqrt_transform(name, data, raise_error=True):
    """
    Apply the `sqrt` transform to `data`.

    Parameters
    ----------
    name : str
        Name of the feature to transform.
    data : numpy array
        Numpy array containing the feature values.
    raise_error : bool, optional
        When set to true, raises an error if the transform is applied to
         a feature that can have negative values.

    Returns
    -------
    new_data : numpy array
        Numpy array containing the transformed feature
        values.

    Raises
    ------
    ValueError
        If the transform is applied to a feature
        that has negative values and `raise_error` is set to true.
    """

    # check if the feature has any negative values
    if np.any(data < 0):
        if raise_error:
            raise ValueError("The sqrt transformation should not be "
                             "applied to feature {} which can have "
                             "negative values".format(name))
        else:
            logging.warning("The sqrt transformation was "
                            "applied to feature {} which has "
                            "negative values for some responses. No system score "
                            "will be generated for such responses".format(name))

    new_data = np.sqrt(data)
    return new_data


def apply_log_transform(name, data, raise_error=True):
    """
    Apply the `log` transform to `data`.

    Parameters
    ----------
    name : str
        Name of the feature to transform.
    data : numpy array
        Numpy array containing the feature values.
    raise_error : bool, optional
        When set to true, raises an error if the transform is applied to
         a feature that has zero or negative values.

    Returns
    -------
    new_data : numpy array
        Numpy array containing the transformed feature
        values.

    Raises
    ------
    ValueError
        If the transform is applied to a feature that
        can be zero or negative and `raise_error` is set to true.
    """

    # check if the feature has any zeros
    if np.any(data == 0):
        if raise_error:
            raise ValueError("The log transformation should not be "
                             "applied to feature {} which can have a "
                             "value of 0".format(name))
        else:
            logging.warning("The log transformation was "
                            "applied to feature {} which has a "
                            "value of 0 for some responses. No system "
                            "score will "
                            "be generated for such responses".format(name))

    # check if the feature has any negative values
    if np.any(data < 0):
        if raise_error:
            raise ValueError("The log transformation should not be "
                             "applied to feature {} which can have "
                             "negative values".format(name))
        else:
            logging.warning("The log transformation was "
                            "applied to feature {} which has "
                            "negative values for some responses. No system "
                            "score will "
                            "be generated for such responses".format(name))

    new_data = np.log(data)
    return new_data


def apply_add_one_inverse_transform(name, data, raise_error=True):
    """
    Apply the add one and invert transform to `data`.

    Parameters
    ----------
    name : str
        Name of the feature to transform.
    data : numpy array
        Numpy array containing the feature values.
    raise_error : bool, optional
        When set to true, raises an error if the transform is applied to
         a feature that has zero or negative values.

    Returns
    -------
    new_data : numpy array
        Numpy array containing the transformed feature
        values.

    Raises
    ------
    ValueError
        If the transform is applied to a feature
        that can be negative and `raise_error` is set to True.
    """

    # check if the feature has any negative values
    if np.any(data < 0):
        if raise_error:
            raise ValueError("The addOneInv transformation should not "
                             "be applied to feature {} which can have "
                             "negative values".format(name))
        else:
            logging.warning("The addOneInv transformation was "
                            "applied to feature {} which has "
                            "negative values for some responses. This can "
                            "change the ranking of the responses".format(name))

    new_data = 1/(data + 1)
    return new_data


def apply_add_one_log_transform(name, data, raise_error=True):
    """
    Apply the add one and log transform to `data`.

    Parameters
    ----------
    name : str
        Name of the feature to transform.
    data : numpy array
        Numpy array containing the feature values.
    raise_error : bool, optional
        When set to true, raises an error if the transform is applied to
         a feature that has zero or negative values.

    Returns
    -------
    new_data : numpy array
        Numpy array that contains the transformed feature
        values.

    Raises
    ------
    ValueError
        If the transform is applied to a feature that
        can be negative.
    """

    # check if the feature has any negative values
    if np.any(data < 0):
        if raise_error:
            raise ValueError("The addOneLn transformation should not "
                             "be applied to feature {} which can have "
                             "negative values".format(name))
        else:
            logging.warning("The log transformation was "
                            "applied to feature {} which has "
                            "negative values for some responses. "
                            "If the feature value remains negative "
                            "after adding one, no score will "
                            "be generated for such responses".format(name))

    new_data = np.log(data + 1)
    return new_data


def transform_feature(name, data, transform, raise_error=True):
    """
    Applies the given transform to all of the values in the given
    numpy array. The values are assumed to be for the feature with
    the given name.

    Parameters
    ----------
    name : str
        Name of the feature to transform.
    data : numpy array
        Numpy array containing the feature values.
    transform : str
        Name of the transform to apply.
    raise_error : bool, optional
        Raise a ValueError if a transformation leads to `Inf` values or may
        change the ranking of the responses

    Returns
    -------
    new_data : numpy array
        Numpy array containing the transformed feature
        values.

    Raises
    ------
    ValueError
        If the given transform is not recognized.

    Note
    ----
    Many of these transformations may be meaningless for features which
    span both negative and positive values. Some transformations may
    throw errors for negative feature values.

    """
    transform_functions = {'inv': apply_inverse_transform,
                           'sqrt': apply_sqrt_transform,
                           'log': apply_log_transform,
                           'addOneInv': apply_add_one_inverse_transform,
                           'addOneLn': apply_add_one_log_transform,
                           'raw': lambda name, data, raise_error: data,
                           'org': lambda name, data, raise_error: data}

    # make sure we have a valid transform function
    if transform is None or transform not in transform_functions:
        raise ValueError('Unrecognized feature transformation: {}'.format(transform))

    transformer = transform_functions.get(transform)
    new_data = transformer(name, data, raise_error)
    return new_data


def preprocess_feature(data,
                       feature_name,
                       feature_transform,
                       feature_mean,
                       feature_sd,
                       exclude_zero_sd=False,
                       raise_transformation_error=True):
    """
    Remove outliers and transform the values in the given numpy array
    using the given outlier and transformation parameters. The values
    are assumed for the given feature name.

    Parameters
    ----------
    data : numpy array
        Numpy array containing the values to pre-process.
    feature_name : str
        Name of the feature being pre-processed.
    feature_transform : str
        Name of the transformation function to apply.
    feature_mean : float
        Mean value to use for outlier detection instead
        of the mean of the given feature values.
    feature_sd : float
        Std. dev. value to use for outlier detection instead
        of the std. dev. of the given feature values.
    exclude_zero_sd : bool, optional
        Check `data` has a zero
        std. dev. Defaults to False.
    raise_transformation_error : bool, optional
        Raise error if any of the transformations lead to inf values
        or may change the ranking of feature values.

    Returns
    -------
    transformed_feature : numpy array
        Numpy array containing the transformed and clamped
        feature values.

    Raises
    ------
    ValueError
        If the given values have zero standard deviation and
        `exclude_zero_sd` is set to `True`.
    """

    # clamp any outlier values that are 4 standard deviations
    # away from the mean
    features_no_outliers = remove_outliers(data, mean=feature_mean, sd=feature_sd)

    # apply the requested transformation to the feature
    transformed_feature = transform_feature(feature_name,
                                            features_no_outliers,
                                            feature_transform,
                                            raise_error=raise_transformation_error)

    # check the standard deviation of the transformed feature
    # we set ddof to 1 so that np.std gave the same result as pandas .std
    # we also set the tolerance limit to account for cases where std
    # is computed as a very low decimal rather than 0
    # We only do this for the training set.
    if exclude_zero_sd:
        feature_sd = np.std(transformed_feature, ddof=1)
        if np.isclose(feature_sd, 0, atol=1e-06):
            raise ValueError("The standard deviation for feature {} "
                             "is 0 after pre-processing. Please exclude "
                             "this feature and re-run the experiment.".format(feature_name))

    return transformed_feature


def preprocess_train_and_test_features(df_train, df_test, feature_specs):
    """
    Pre-process those features in the given training and testing
    data frame `df` whose specifications are contained in
    `feature_specs`. Also return a third data frame containing the
    feature specs themselves.

    Parameters
    ----------
    df_train : pandas DataFrame
        Data frame containing the raw feature values
        for the training set.
    df_test : pandas DataFrame
        Data frame containing the raw feature values
        for the test set.
    feature_specs : dict
        Dictionary containing the various specifications
        from the feature JSON file.
    """

    # keep the original data frames and make copies
    # that only include features used in the model
    df_train_preprocessed = df_train.copy()
    df_test_preprocessed = df_test.copy()

    # we also need to create a data frame that includes
    # all relevant information about each feature
    df_feature_info = pd.DataFrame()

    # now iterate over each feature
    for fdict in feature_specs['features']:

        feature_name = fdict['feature']
        feature_transformation = fdict['transform']
        feature_sign = fdict['sign']

        train_feature_mean = df_train[feature_name].mean()
        train_feature_sd = df_train[feature_name].std()

        training_feature_values = df_train[feature_name].values
        df_train_preprocessed[feature_name] = preprocess_feature(training_feature_values,
                                                                 feature_name,
                                                                 feature_transformation,
                                                                 train_feature_mean,
                                                                 train_feature_sd,
                                                                 exclude_zero_sd=True)

        testing_feature_values = df_test[feature_name].values
        df_test_preprocessed[feature_name] = preprocess_feature(testing_feature_values,
                                                                feature_name,
                                                                feature_transformation,
                                                                train_feature_mean,
                                                                train_feature_sd)

        # Standardize the features using the mean and sd computed on the
        # training set. These are computed separately because we need to
        # get the mean of transformed feature before standardization.
        train_transformed_mean = df_train_preprocessed[feature_name].mean()
        train_transformed_sd = df_train_preprocessed[feature_name].std()

        df_train_preprocessed[feature_name] = (df_train_preprocessed[feature_name] - train_transformed_mean) / train_transformed_sd
        df_test_preprocessed[feature_name] = (df_test_preprocessed[feature_name] - train_transformed_mean) / train_transformed_sd

        # Multiply both train and test feature by weight. Within the
        # current SR timeline, the mean of the transformed train
        # feature used to standardize test features has to be
        # computed before multiplying the train feature by the weight.
        df_train_preprocessed[feature_name] = df_train_preprocessed[feature_name] * feature_sign
        df_test_preprocessed[feature_name] = df_test_preprocessed[feature_name] * feature_sign

        # update the feature preprocessing metadata frame
        df_feature = pd.DataFrame([{"feature": feature_name,
                                    "transform": feature_transformation,
                                    "sign": feature_sign,
                                    "train_mean": train_feature_mean,
                                    "train_sd": train_feature_sd,
                                    "train_transformed_mean": train_transformed_mean,
                                    "train_transformed_sd": train_transformed_sd}])
        df_feature_info = df_feature_info.append(df_feature)

    # reset the index for the feature metadata frame
    # since we built it up row by row
    df_feature_info = df_feature_info.reset_index().drop('index', 1)

    # return the three data frames
    return (df_train_preprocessed,
            df_test_preprocessed,
            df_feature_info)
