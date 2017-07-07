"""
Functions dealing with preprocessing input data.

:author: Nitin Madnani (nmadnani@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:organization: ETS
"""

import logging

import numpy as np
import pandas as pd

from skll.data import safe_float as string_to_number

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

    flag_columns = list(flag_column_dict.keys())

    if not flag_columns:
        return df.copy(), pd.DataFrame(columns=df.columns)
    else:
        # check that all columns are present
        missing_flag_columns = set(flag_columns).difference(df.columns)
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
        # we convert all strings and integers to floats such that, for
        # example, “1”, 1, and “1.0" all map to 1.0. To do this, we will
        # first convert all the strings to numbers and then convert
        # all the integers to floats.
        int_to_float = lambda x: float(x) if type(x) == int else x
        convert_to_float = lambda x: int_to_float(string_to_number(x))
        flag_column_dict_to_float = {key: list(map(convert_to_float, value))
                                     for (key, value) in flag_column_dict.items()}

        # and now convert the the values in the feature column in the data frame
        df_new = df[flag_columns].copy()
        df_new = df_new.applymap(convert_to_float)

        # identify responses with values which satisfy the condition
        full_mask = df_new.isin(flag_column_dict_to_float)
        flagged_mask = full_mask[list(flag_column_dict_to_float.keys())].all(1)

        # return the columns from the original frame that was passed in
        # so that all data types remain the same and are not changed
        df_responses_with_requested_flags = df[flagged_mask].copy()
        df_responses_with_excluded_flags = df[~flagged_mask].copy()

        # make sure that the remaining data frame is not empty
        if len(df_responses_with_requested_flags) == 0:
            raise ValueError("No responses remaining after filtering "
                             "on flag columns. No further analysis can "
                             "be run.")

        # reset the index
        df_responses_with_requested_flags.reset_index(drop=True, inplace=True)
        df_responses_with_excluded_flags.reset_index(drop=True, inplace=True)

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
    # We also filter out inf values. Since these can only be generated
    # during transformations we convert them to NaNs for consistency.
    bad_rows = df_filtered[df_filtered[column].isnull() | np.isinf(df_filtered[column])]

    # drop the NaNs that we might have gotten
    df_filtered = df_filtered[df_filtered[column].notnull() & ~np.isinf(df_filtered[column])]

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
        if np.isclose(feature_sd, 0, atol=1e-07):
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
            logging.warning("The inverse transformation was applied to "
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

    with np.errstate(divide='ignore'):
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

    with np.errstate(invalid='ignore'):
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
        if np.isclose(feature_sd, 0, atol=1e-07):
            raise ValueError("The standard deviation for feature {} "
                             "is 0 after pre-processing. Please exclude "
                             "this feature and re-run the experiment.".format(feature_name))

    return transformed_feature


def preprocess_train_and_test_features(df_train, df_test, df_feature_specs):
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
    df_feature_specs : pandas DataFrame
        Data frame containing the various specifications
        from the feature file.
    """

    # keep the original data frames and make copies
    # that only include features used in the model
    df_train_preprocessed = df_train.copy()
    df_test_preprocessed = df_test.copy()

    # we also need to create a data frame that includes
    # all relevant information about each feature
    df_feature_info = pd.DataFrame()

    # make feature the index of df_feature_specs
    df_feature_specs.index = df_feature_specs['feature']

    # now iterate over each feature
    for feature_name in df_feature_specs['feature']:

        feature_transformation = df_feature_specs.get_value(feature_name, 'transform')
        feature_sign = df_feature_specs.get_value(feature_name, 'sign')

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

        # Multiply both train and test feature by sign.
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


def preprocess_new_data(df_input,
                        df_feature_info):

    """
    Process a data frame with feature values by applying
    :ref:`preprocessing parameters <preprocessing_parameters>`
    stored in `df_feature_info`.

    Parameters
    ----------

    df_input : pandas DataFrame
        Data frame with raw feature values that will be used to generate
        the scores. Each feature is stored in a separate column. Each row
        corresponds to one response. There should also be a column named
        `spkitemid` containing a unique ID for each response.

    df_feature_info : pandas DataFrame
        Data frame with preprocessing parameters stored in the following columns:

            - `feature` : the name of the feature. These should match the feature names in `df_input`.
            - `sign` : `1` or `-1`.  Indicates whether the feature value needs to be multiplied by -1.
            - `transform` : :ref:`transformation <json_transformation>` that needs to be applied to this feature
            - `train_mean`, `train_sd` : mean and standard deviation for outlier truncation.
            - `train_transformed_mean`,`train_transformed_sd` : mean and standard deviation for computing `z`-scores.

    Returns
    -------
    df_features_preprocessed : pandas DataFrame
        Data frame with processed feature values

    df_excluded: pandas DataFrame
        Data frame with responses excluded from further analysis
        due to non-numeric feature values in the original file
        or after applying transformations. The data frame always contains the
        original feature values.


    Raises
    ------
    KeyError :
        if some of the features specified in `df_feature_info` are not present in `df_input`

    ValueError :
        if all responses have at least one non-numeric feature value and therefore no score can be generated for any of the responses.
    """

    logger = logging.getLogger(__name__)

    # get the list of required features

    required_features = df_feature_info.index.tolist()

    # ensure that all the features that are needed by the model
    # are present in the input file
    input_feature_columns = [c for c in df_input if c != 'spkitemid']
    missing_features = set(required_features).difference(input_feature_columns)
    if missing_features:
        raise KeyError('The input feature file is missing the following features: {}'.format(missing_features))

    extra_features = set(input_feature_columns).difference(required_features + ['spkitemid'])
    if extra_features:
        logging.warning('The following extraenous features will be ignored: {}'.format(extra_features))

    # keep the required features plus the id
    features_to_keep = ['spkitemid'] + required_features

    # check if actually have the human scores for this data and add
    # sc1 to preprocessed features for consistency with other tools
    has_human_scores = 'sc1' in df_input
    if has_human_scores:
        features_to_keep.append('sc1')

    df_features = df_input[features_to_keep]

    # preprocess the feature values
    logger.info('Pre-processing input features')

    # first we need to filter out NaNs and any other
    # weird features, the same way we did for rsmtool.
    df_filtered = df_features.copy()
    df_excluded = pd.DataFrame(columns=df_filtered.columns)

    for feature_name in required_features:
        newdf, newdf_excluded = filter_on_column(df_filtered, feature_name, 'spkitemid',
                                                 exclude_zeros=False,
                                                 exclude_zero_sd=False)
        del df_filtered
        df_filtered = newdf
        with np.errstate(divide='ignore'):
            df_excluded = pd.merge(df_excluded, newdf_excluded, how='outer')

    # make sure that the remaining data frame is not empty
    if len(df_filtered) == 0:
        raise ValueError("There are no responses left after "
                         "filtering out non-numeric feature values. No analysis "
                         "will be run")

    df_features = df_filtered.copy()
    df_features_preprocessed = df_features.copy()
    for feature_name in required_features:

        feature_values = df_features_preprocessed[feature_name].values

        feature_transformation = df_feature_info.loc[feature_name]['transform']
        feature_sign = df_feature_info.loc[feature_name]['sign']

        train_feature_mean = df_feature_info.loc[feature_name]['train_mean']
        train_feature_sd = df_feature_info.loc[feature_name]['train_sd']

        train_transformed_mean = df_feature_info.loc[feature_name]['train_transformed_mean']
        train_transformed_sd = df_feature_info.loc[feature_name]['train_transformed_sd']

        # transform the feature values and remove outliers
        df_features_preprocessed[feature_name] = preprocess_feature(feature_values,
                                                                    feature_name,
                                                                    feature_transformation,
                                                                    train_feature_mean,
                                                                    train_feature_sd,
                                                                    exclude_zero_sd=False,
                                                                    raise_transformation_error=False)

        # filter the feature values once again to remove possible NaN and inf values that
        # might have emerged when applying transformations.
        # We do not need to do that if no transformation was applied.
        if not feature_transformation in ['raw', 'org']:
            # check that there are indeed inf or Nan values
            if np.isnan(df_features_preprocessed[feature_name]).any() or \
               np.isinf(df_features_preprocessed[feature_name]).any():
                    newdf, newdf_excluded = filter_on_column(df_features_preprocessed, feature_name, 'spkitemid',
                                                             exclude_zeros=False,
                                                             exclude_zero_sd=False)
                    del df_features_preprocessed
                    df_features_preprocessed = newdf
                    # add the response(s) with missing values to the excluded responses
                    # but make sure we are adding the original values, not the preprocessed
                    # ones
                    newdf_excluded_original = df_features[df_features['spkitemid'].isin(newdf_excluded['spkitemid'])].copy()
                    df_excluded = pd.merge(df_excluded, newdf_excluded_original, how='outer')

        # now standardize the feature values
        df_features_preprocessed[feature_name] = (df_features_preprocessed[feature_name] - train_transformed_mean) / train_transformed_sd

        # Multiply features by sign.
        df_features_preprocessed[feature_name] = df_features_preprocessed[feature_name] * feature_sign

    return (df_features_preprocessed, df_excluded)
