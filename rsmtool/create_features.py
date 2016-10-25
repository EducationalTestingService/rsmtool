"""
Functions dealing with generating feature names and transformations if there is no feature config file.

:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:organization: ETS
"""

import logging

import numpy as np
import pandas as pd

from scipy.stats.stats import pearsonr

from rsmtool.preprocess import transform_feature


def select_by_subset(feature_columns, feature_subset_specs, subset):

    subset_features = feature_subset_specs[feature_subset_specs[subset] == 1]['Feature']
    feature_names = [feature for feature in feature_columns if feature in subset_features.values]

    # check whether there are any features in the data file and raise warning
    if len(feature_columns) != len(feature_names):
        extra_columns = set(feature_columns).difference(set(feature_subset_specs['Feature']))
        if extra_columns:
            logging.warning("No subset information was available for the "
                            "following columns in the input file. These "
                            "columns will not be used in the model: "
                            "{}".format(', '.join(extra_columns)))
    if len(subset_features) != len(feature_names):
        extra_subset_features = set(subset_features).difference(set(feature_names))
        if extra_subset_features:
            logging.warning("The following features were included into the {} "
                            "subset in the feature_subset_file but were not "
                            "specified in the input data: {}".format(subset,
                                                                     ', '.join(extra_subset_features)))
    return feature_names


def select_by_prefix(feature_columns, prefixes):

    feature_names = []
    split_names = [name.split('\t') for name in feature_columns]

    for prefix in prefixes:
        selected_features = [feature_columns[i] for i in range(len(split_names)) if split_names[i][0] == prefix]
        feature_names.extend(selected_features)
        if len(selected_features) == 0:
            logging.warning("No feature names match prefix {}".format(prefix))

    if (len(feature_names)) == 0:
        raise ValueError("There are no feature names matching any of the prefixes in {}. Please make sure that the prefixes are specified correctly and are separated by \t".format((', ').join(prefixes)))
    return feature_names


def generate_feature_names(df,
                           reserved_column_names,
                           feature_subset_specs,
                           feature_subset,
                           feature_prefix):

    """
    Generate the feature names from the column
    names of the given data frame and select the
    specified subset of features.
    """

    # Exclude the reserved names
    possible_feature_names = [cname for cname in df.columns
                              if cname not in reserved_column_names]

    # Select the features by subset or prefix.
    # In the future, we may add option to do
    # both if there is a use case.
    if feature_subset is not None and feature_prefix is not None:
        raise ValueError("It is not possible to select feature "
                         "subset by using both feature_subset and "
                         "feature_prefix. Please remove one of "
                         "these fields.")

    if feature_subset is not None:
        feature_names = select_by_subset(possible_feature_names,
                                         feature_subset_specs,
                                         feature_subset)
    elif feature_prefix is not None:
            feature_names = select_by_prefix(possible_feature_names, feature_prefix)
    else:
        feature_names = possible_feature_names

    return feature_names


def generate_default_specs(feature_names):

    """
    Generate default feature "specifications" for the features
    with the given names. The specifications are stored as a data frame with 
    three columns "feature", "transform", and "sign".

    Parameters
    ----------
    feature_names: list
        List of feature names for which to generate specifications.

    Returns
    -------
    feature_specs: pandas DataFrame
        A dataframe with feature specifications that can be saved as a
        :ref:`feature list file <example_feature_csv>`.

    Note
    ----
    Since these are default specifications, the values for the
    `transform` column for each feature will be `"raw"` and the value
    for the `sign` column will be `1`.

    """

    df_feature_specs = pd.DataFrame({'feature': feature_names})
    df_feature_specs['transform'] = 'raw'
    df_feature_specs['sign'] = 1.0

    return df_feature_specs


def find_feature_sign(feature, sign_dict):

    """
    Get the sign from the feature.csv file
    """

    logger = logging.getLogger(__name__)

    if not feature in sign_dict.keys():
        logger.warning("No information about sign is available for feature {}. "
                       "The feature will be assigned the default positive weight.".format(feature))
        feature_sign_numeric = 1.0
    else:
        feature_sign_string = sign_dict[feature]
        feature_sign_numeric = -1.0 if feature_sign_string == '-' else 1.0
    return feature_sign_numeric


def find_feature_transformation(feature_name, feature_value, scores):

    """
    Identify the best transformation based on the
    highest absolute Pearson correlation with human score.

    Parameters
    ----------
    feature_name: str
        Name of feature for which to find the transformation.

    feature_value: pandas Series
        Series containing feature values.

    scores: pandas Series
        Numeric human scores.

    Returns
    -------
    best_transformation: str
        The name of the transformation which gives the highest correlation
        between the feature values and the human scores. See
        :ref:`documentation <select_transformations_rsmtool>` for the
        full list of transformations.
    """

    # Do not use sqrt and ln for potential negative features.
    # Do not use inv for positive features.
    if any(feature_value < 0):
        applicable_transformations = ['org', 'inv']
    else:
        applicable_transformations = ['org', 'sqrt', 'addOneInv', 'addOneLn']

    correlations = []
    for trans in applicable_transformations:
        try:
            transformed_value = transform_feature(feature_name, feature_value, trans)
            correlations.append(abs(pearsonr(transformed_value, scores)[0]))
        except ValueError:
            # If the transformation returns an error, append 0.
            correlations.append(0)
    best = np.argmax(correlations)
    best_transformation = applicable_transformations[best]
    return best_transformation


def generate_specs_from_data(feature_names,
                             train_label,
                             df_train,
                             feature_subset_specs=None,
                             feature_sign=None):

    """
    Generate feature specifications using the features.csv
    for sign and the correlation with score to identify
    the best transformation.
    """

    # get feature sign info if available
    if feature_sign:
        # Convert to dictionary {feature:sign}
        sign_dict = dict(zip(feature_subset_specs.Feature,
                             feature_subset_specs['Sign_{}'.format(feature_sign)]))
    # else create an empty dictionary
    else:
        sign_dict = {}

    feature_specs = []
    feature_dict = {}
    for feature in feature_names:
        feature_dict['feature'] = feature
        feature_dict['transform'] = find_feature_transformation(feature, df_train[feature], df_train[train_label])
        feature_dict['sign'] = find_feature_sign(feature, sign_dict)

        # Change the sign for inverse and addOneInv transformations
        if feature_dict['transform'] in ['inv', 'addOneInv']:
            feature_dict['sign'] = feature_dict['sign'] * -1
        feature_specs.append(feature_dict)
        feature_dict = {}

    df_feature_specs = pd.DataFrame(feature_specs)
    return df_feature_specs
