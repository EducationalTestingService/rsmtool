"""
Classes for preprocessing input data in various contexts.

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

import logging
import re
import warnings

import numpy as np
import pandas as pd

from collections import defaultdict

from numpy.random import RandomState

from .container import DataContainer
from .reader import DataReader
from .reporter import Reporter
from .transformer import FeatureTransformer
from .utils.conversion import convert_to_float
from .utils.models import is_built_in_model, is_skll_model


class FeatureSubsetProcessor:
    """
    Encapsulate feature sub-setting methods.
    """

    @classmethod
    def select_by_subset(cls, feature_columns, feature_subset_specs, subset):
        """
        Select feature columns using feature subset specs.

        Parameters
        ----------
        feature_columns : list
            A list of feature columns
        feature_subset_specs : pd.DataFrame
            The feature subset spec DataFrame.
        subset : str
            The column to subset.

        Returns
        -------
        feature_names : list
            A list of feature names to include.
        """

        feature_subset = feature_subset_specs[feature_subset_specs[subset] == 1]['Feature']
        feature_names = [feature for feature in feature_columns
                         if feature in feature_subset.values]

        # check whether there are any features in the data file and raise warning
        if len(feature_columns) != len(feature_names):
            feature_subset_specs_set = set(feature_subset_specs['Feature'])
            extra_columns = set(feature_columns).difference(feature_subset_specs_set)
            if extra_columns:
                logging.warning("No subset information was available for the "
                                "following columns in the input file. These "
                                "columns will not be used in the model: "
                                "{}".format(', '.join(extra_columns)))
        if len(feature_subset) != len(feature_names):
            extra_subset_features = set(feature_subset).difference(set(feature_names))
            if extra_subset_features:
                logging.warning("The following features were included into the {} "
                                "subset in the feature_subset_file but were not "
                                "specified in the input data: "
                                "{}".format(subset, ', '.join(extra_subset_features)))
        return feature_names

    @classmethod
    def check_feature_subset_file(cls, df, subset=None, sign=None):
        """
        Check that the file is in the correct format and contains all
        the requested values. Raises an exception if it finds any errors
        but otherwise returns nothing.

        Parameters
        ----------
        df : pd.DataFrame
            The feature subset file DataFrame.
        subset : str, optional
            Name of a pre-defined feature subset.
            Defaults to None.
        sign : str, optional
            Value of the sign
            Defaults to None.

        Raises
        ------
        ValueError
            If any columns are missing from the subset file
            or if any of the columns contain invalid values.
        """

        # we want to allow title-cased names of columns for historical reasons
        # e.g., `Feature` instead of `feature` etc.

        df_feature_specs = df.copy()

        if ('feature' not in df_feature_specs and
                'Feature' not in df_feature_specs):
            raise ValueError("The feature_subset_file must contain "
                             "a column named 'feature' "
                             "containing the feature names.")
        if subset:
            if subset not in df_feature_specs:
                raise ValueError("Unknown value for feature_subset: {}".format(subset))

            if not df_feature_specs[subset].isin([0, 1]).all():
                raise ValueError("The subset columns in feature "
                                 "file can only contain 0 or 1")

        if sign:
            possible_sign_columns = ['sign_{}'.format(sign),
                                     'Sign_{}'.format(sign)]
            existing_sign_columns = [c for c in possible_sign_columns
                                     if c in df_feature_specs]
            if len(existing_sign_columns) > 1:
                raise ValueError("The feature_subset_file contains "
                                 "multiple columns for sign: "
                                 "{}".format(' ,'.join(existing_sign_columns)))
            elif len(existing_sign_columns) == 0:
                raise ValueError("The feature_subset_file must "
                                 "contain the requested "
                                 "sign column 'sign_{}'".format(sign))
            else:
                sign_column = existing_sign_columns[0]

            if not df_feature_specs[sign_column].isin(['-', '+']).all():
                raise ValueError("The sign columns in feature "
                                 "file can only contain - or +")


class FeatureSpecsProcessor:
    """
    Encapsulate feature file processing methods.
    """

    @classmethod
    def generate_default_specs(cls, feature_names):
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

    @classmethod
    def find_feature_sign(cls, feature, sign_dict):
        """
        Get the sign from the feature.csv file

        Parameters
        ----------
        feature : str
            The name of the feature
        sign_dict : dict
            A dictionary of feature signs.

        Returns
        -------
        feature_sign_numeric : float
            The signed feature.
        """

        if feature not in sign_dict.keys():
            logging.warning("No information about sign is available "
                            "for feature {}. The feature will be assigned "
                            "the default positive weight.".format(feature))
            feature_sign_numeric = 1.0
        else:
            feature_sign_string = sign_dict[feature]
            feature_sign_numeric = -1.0 if feature_sign_string == '-' else 1.0
        return feature_sign_numeric

    @classmethod
    def validate_feature_specs(cls, df, use_truncations=False):
        """
        Check the supplied feature specs to make sure that there are no duplicate
        feature names and that all columns are in the right format. Add the default values
        for  `transform` and `sign` if none is supplied

        Parameters
        ----------
        df : pd.DataFrame
            The feature specification DataFrame to validate.
        use_truncations : bool, optional
            Whether to use truncation values. If this is
            True and truncation values are not specified,
            raise an error.
            Defaults to False.

        Returns
        ------
        df_specs_new : pandas DataFrame
                A data frame with normalized values

        Raises
        ------
        KeyError :
            If the data frame does not have a ``feature`` column.
        ValueError:
            If there are duplicate values in the ``feature`` column
            or if the ``sign`` column contains invalid values.
        ValueError
            If ``use_truncations`` is set to True, and no
            ``min`` and ``max`` columns exist in the data set.
        """
        df_specs_org = df
        df_specs_new = df_specs_org.copy()

        expected_columns = ['feature', 'sign', 'transform']

        # we allow internally the use of 'Feature' since
        # this is the column name in subset_feature_file.
        if 'Feature' in df_specs_org:
            df_specs_new['feature'] = df_specs_org['Feature']

        # check that we have a column named `feature`
        if 'feature' not in df_specs_new:
            raise KeyError("The feature file must contain a "
                           "column named 'feature'")

        # check to make sure that there are no duplicate feature names
        feature_name_count = df_specs_new['feature'].value_counts()
        duplicate_features = feature_name_count[feature_name_count > 1]
        if len(duplicate_features) > 0:
            raise ValueError("The following feature names "
                             " are duplicated in the feature "
                             "file: {}".format(duplicate_features.index))

        # if we have `sign` column, check that it can be converted to float
        if 'sign' in df_specs_new:
            try:
                df_specs_new['sign'] = df_specs_new['sign'].astype(float)
                assert np.all(df_specs_new['sign'].isin([-1, 1]))
            except (ValueError, AssertionError):
                raise ValueError("The `sign` column in the feature"
                                 "file can only contain '1' or '-1'")
        else:
            df_specs_new['sign'] = 1

        if 'transform' not in df_specs_new:
            df_specs_new['transform'] = 'raw'

        if use_truncations:
            if not all(col in df_specs_new for col in ['min', 'max']):
                raise ValueError('The ``use_truncation_thresholds`` configuration option '
                                 'was specified, but no ``min`` or ``max`` columns exist '
                                 'in the feature file.')

            # add ``min`` and ``max`` to the
            # list of expected columns
            expected_columns.extend(['min', 'max'])

        df_specs_new = df_specs_new[expected_columns]
        return df_specs_new

    @classmethod
    def generate_specs(cls,
                       df,
                       feature_names,
                       train_label,
                       feature_subset=None,
                       feature_sign=None):
        """
        Generate feature specifications using the features.csv
        for sign and the correlation with score to identify
        the best transformation.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame form which to generate specs.
        feature_names : list
            A list of feature names.
        train_label : str
            The label column for the training data
        feature_subset : pd.DataFrame, optional
            A feature_subset_specs DataFrame
        feature_sign : int, optional
            The sign of the feature.

        Returns
        -------
        df_feature_specs : pd.DataFrame
            A feature specifications DataFrame
        """

        # get feature sign info if available
        if feature_sign:
            # Convert to dictionary {feature:sign}
            sign_dict = dict(zip(feature_subset.Feature,
                                 feature_subset['Sign_{}'.format(feature_sign)]))
        # else create an empty dictionary
        else:
            sign_dict = {}

        feature_specs = []
        feature_dict = {}
        for feature in feature_names:
            feature_dict['feature'] = feature
            feature_dict['transform'] = FeatureTransformer.find_feature_transform(feature,
                                                                                  df[feature],
                                                                                  df[train_label])

            feature_dict['sign'] = FeatureSpecsProcessor.find_feature_sign(feature, sign_dict)

            # Change the sign for inverse and addOneInv transformations
            if feature_dict['transform'] in ['inv', 'addOneInv']:
                feature_dict['sign'] = feature_dict['sign'] * -1

            feature_specs.append(feature_dict)
            feature_dict = {}

        df_feature_specs = pd.DataFrame(feature_specs)
        return df_feature_specs


class FeaturePreprocessor:
    """
    A class to pre-process training and testing features.
    """

    @staticmethod
    def check_model_name(model_name):
        """
        Check that the given model name is valid and determine its type.

        Parameters
        ----------
        model_name : str
            Name of the model.

        Returns
        -------
        model_type: str
            One of `BUILTIN` or `SKLL`.

        Raises
        ------
        ValueError
            If the model is not supported.
        """

        if is_built_in_model(model_name):
            model_type = 'BUILTIN'
        elif is_skll_model(model_name):
            model_type = 'SKLL'
        else:
            raise ValueError("The specified model {} "
                             "was not found. Please "
                             "check the spelling.".format(model_name))

        return model_type

    @staticmethod
    def trim(values,
             trim_min,
             trim_max,
             tolerance=0.4998):
        """
        Trim the values contained in the given numpy array to
        `trim_min` - `tolerance` as the floor and
        `trim_max` + `tolerance` as the ceiling.

        Parameters
        ----------
        values : list or np.array
            The values to trim.
        trim_min : float
            The lowest score on the score point, used for
            trimming the raw regression predictions.
        trim_max : float
            The highest score on the score point, used for
            trimming the raw regression predictions.
        tolerance : float, optional
            The tolerance that will be used to compute the
            trim interval. Defaults to 0.4998.

        Returns
        -------
        trimmed_values : np.array
            Trimmed values.
        """
        if isinstance(values, list):
            values = np.array(values)

        new_max = trim_max + tolerance
        new_min = trim_min - tolerance
        trimmed_values = values.copy()
        trimmed_values[trimmed_values > new_max] = new_max
        trimmed_values[trimmed_values < new_min] = new_min
        return trimmed_values

    @staticmethod
    def remove_outliers(values,
                        mean=None,
                        sd=None,
                        sd_multiplier=4):
        """
        Clamp any values in the given numpy array that are
        +/- `sd_multiplier` (:math:`m`) standard deviations (:math:`\\sigma`)
        away from the mean (:math:`\\mu`). Use given `mean` and `sd` instead
        of computing :math:`\\sigma` and :math:`\\mu`, if specified.
        The values are clamped to the interval .. math::

            [\\mu - m * \\sigma, \\mu + m * \\sigma]

        Parameters
        ----------
        values : np.array
            The values from which to remove outliers.
        mean : int or float, optional
            Use the given mean value when computing outliers
            instead of the mean from the data.
            Defaults to None
        sd : None, optional
            Use the given std. dev. value when computing
            outliers instead of the std. dev. from the
            data.
            Defaults to None.
        sd_multiplier : int, optional
            Use the given multipler for the std. dev. when
            computing the outliers. Defaults to 4.
            Defaults to 4.

        Returns
        -------
        new_values : np.array
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

    @staticmethod
    def remove_outliers_using_truncations(values,
                                          feature_name,
                                          truncations):
        """
        Remove outliers using truncation groups,
        rather than calculating the outliers based
        on the training set.

        Parameters
        ----------
        values : np.array
            The values from which to remove outliers.
        feature_name : str
            Name of the feature whose outliers are
            being clamped.
        truncations : pd.DataFrame
            A data frame with truncation values. The
            features should be set as the index.

        Returns
        -------
        new_values : numpy array
            Numpy array with the outliers clamped.
        """

        # convert data to a numpy float array before doing any clamping
        new_values = np.array(values, dtype=np.float)

        minimum = truncations.loc[feature_name, 'min']
        maximum = truncations.loc[feature_name, 'max']

        new_values[new_values > maximum] = maximum
        new_values[new_values < minimum] = minimum
        return new_values

    @staticmethod
    def select_candidates(df,
                          N,
                          candidate_col='candidate'):
        """
        Only select candidates which have responses to N or more items.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame from which to select candidates with N or more items.
        N: int
            minimal number of items per candidate
        candidate_col : str, optional
            name of the column which contains candidate ids.
            Defaults to 'candidate'.

        Returns
        -------
        df_included: pandas DataFrame
            Data frame with responses from candidates with responses to N
            or more items
        df_excluded: pandas DataFrame
            Data frame with responses from candidates with responses to
            less than N items
        """

        items_per_candidate = df[candidate_col].value_counts()

        selected_candidates = items_per_candidate[items_per_candidate >= N]
        selected_candidates = selected_candidates.index

        df_included = df[df[candidate_col].isin(selected_candidates)].copy()
        df_excluded = df[~df[candidate_col].isin(selected_candidates)].copy()

        # reset indices
        df_included.reset_index(drop=True, inplace=True)
        df_excluded.reset_index(drop=True, inplace=True)

        return (df_included,
                df_excluded)

    @staticmethod
    def check_subgroups(df, subgroups):
        """
        Check that all subgroups, if specified, correspond to columns in the
        provided data frame, and replace all NaNs in subgroups values with
        'No info' for later convenience. Raises an exception if any specified
        subgroup columns are missing.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with subgroups to check.
        subgroups : list of str
            List of column names that contain grouping
            information.

        Returns
        -------
        df : pandas DataFrame
             Modified input data frame with NaNs replaced.

        Raises
        ------
        KeyError
            If the data does not contain columns for all subgroups
        """

        missing_sub_cols = set(subgroups).difference(df.columns)
        if missing_sub_cols:
            raise KeyError("The data does not contain columns "
                           "for all subgroups specified in the "
                           "configuration file. Please check for "
                           "capitalization and other spelling "
                           "errors and make sure the subgroup "
                           "names do not contain hyphens. "
                           "The data does not have columns "
                           "for the following "
                           "subgroups: {}".format(', '.join(missing_sub_cols)))

        # replace any empty values in subgroups values by "No info"
        empty_value = re.compile(r"^\s*$")
        df[subgroups] = df[subgroups].replace(to_replace=empty_value,
                                              value='No info')
        return df

    @staticmethod
    def rename_default_columns(df,
                               requested_feature_names,
                               id_column,
                               first_human_score_column,
                               second_human_score_column,
                               length_column,
                               system_score_column,
                               candidate_column):
        """
        Standardize all column names and rename all columns with default
        names to ##NAME##.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame whose columns to rename.
        requested_feature_names : list
            List of feature column names that we want
            to include in the scoring model.
        id_column : str
            Column name containing the response IDs.
        first_human_score_column : str or None
            Column name containing the H1 scores.
        second_human_score_column : str or None
            Column name containing the H2 scores.
            Should be None if no H2 scores are available.
        length_column : str or None
            Column name containing response lengths.
            Should be None if lengths are not available.
        system_score_column : str
            Column name containing the score predicted
            by the system. This is only used for RSMEval.
        candidate_column : str or None
            Column name containing identifying information
            at the candidate level. Should be None if such
            information is not available.

        Returns
        -------
        df : pandas DataFrame
            Modified input data frame with all the approximate
            re-namings.
        """

        df = df.copy()

        columns = [id_column,
                   first_human_score_column,
                   second_human_score_column,
                   length_column,
                   system_score_column,
                   candidate_column]

        defaults = ['spkitemid', 'sc1', 'sc2', 'length', 'raw', 'candidate']

        # create a dictionary of name mapping for used columns
        name_mapping = dict(filter(lambda t: t[0] is not None, zip(columns,
                                                                   defaults)))

        # find the columns where the names match the default names
        correct_defaults = [column for (column, default)
                            in name_mapping.items()
                            if column == default]

        # find the columns with default names reserved for other columns
        # which are not used as features in the model
        columns_with_incorrect_default_names = [column for column in df.columns
                                                if (column in defaults and
                                                    column not in correct_defaults and
                                                    column not in requested_feature_names)]
        # rename these columns
        if columns_with_incorrect_default_names:
            new_column_names = ['##{}##'.format(column) for column
                                in columns_with_incorrect_default_names]
            df.rename(columns=dict(zip(columns_with_incorrect_default_names,
                                       new_column_names)),
                      inplace=True)

        # find the columns where the names do not match the default
        columns_with_custom_names = [column for column in name_mapping
                                     if column not in correct_defaults]

        # rename the custom-named columns to default values
        for column in columns_with_custom_names:

            # if the column has already been renamed because it used a
            # default name, then use the updated name
            if column in columns_with_incorrect_default_names:
                df.rename(columns={'##{}##'.format(column):
                                   name_mapping[column]},
                          inplace=True)
            else:
                df.rename(columns={column:
                                   name_mapping[column]},
                          inplace=True)

        return df

    @staticmethod
    def filter_on_column(df,
                         column,
                         id_column,
                         exclude_zeros=False,
                         exclude_zero_sd=False):
        """
        Filter out the rows in the given data frame that contain non-numeric
        (or zero, if specified) values in the specified column. Additionally,
        it may exclude any columns if they have a standard deviation
        (:math:`\\sigma`) of 0.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to filter on.
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

        # create a copy of the original data frame
        df_filter = df.copy()

        # we start out assuming that we will not drop this column
        drop_column = False

        # return a copy of the original data frame if
        # the given column does not exist at all
        if column not in df.columns:
            return df_filter

        # Force convert the label column to numeric and
        # convert whatever can't be converted to a NaN
        df_filter[column] = pd.to_numeric(df_filter[column],
                                          errors='coerce').astype(float)

        # Save the values that have been converted to NaNs
        # as a separate data frame. We want to keep them as NaNs
        # to do more analyses later. We also filter out inf values.
        # Since these can only be generated during transformations,
        # we include them with NaNs for consistency.
        bad_rows = df_filter[column].isnull() | np.isinf(df_filter[column])
        df_bad_rows = df_filter[bad_rows]

        # if the column contained only non-numeric values, we need to drop it
        if len(df_bad_rows) == len(df_filter):
            logging.info(f"Feature {column} was excluded from the model "
                         f"because it only contains non-numeric values.")
            drop_column = True

        # now drop the above bad rows containing NaNs from our data frame
        df_filter = df_filter[~bad_rows]

        # exclude zeros if specified
        if exclude_zeros:
            zero_rows = df_filter[column] == 0
            df_zero_rows = df_filter[zero_rows]
            df_filter = df_filter[~zero_rows]
        else:
            df_zero_rows = pd.DataFrame()

        # combine all the filtered rows into a single data frame
        df_exclude = pd.concat([df_bad_rows, df_zero_rows], sort=True)

        # reset the index so that the indexing works correctly
        # for the next feature with missing values
        df_filter.reset_index(drop=True, inplace=True)
        df_exclude.reset_index(drop=True, inplace=True)

        # Check if the the standard deviation equals zero:
        # for training set sd == 0 will break normalization.
        # We set the tolerance level to the 6th digit
        # to account for the possibility that the exact value
        # computed by `std()` is not 0
        if exclude_zero_sd is True:
            feature_sd = df_filter[column].std()
            if np.isclose(feature_sd, 0, atol=1e-07):
                logging.info(f"Feature {column} was excluded from the model "
                             f"because its standard deviation in the "
                             f"training set is equal to 0.")
                drop_column = True

        # if `drop_column` is true, then we need to drop the column
        if drop_column:
            df_filter = df_filter.drop(column, axis=1)
            df_exclude = df_exclude.drop(column, axis=1)

        # return the filtered rows and the new data frame
        return (df_filter, df_exclude)

    @staticmethod
    def process_predictions(df_test_predictions,
                            train_predictions_mean,
                            train_predictions_sd,
                            human_labels_mean,
                            human_labels_sd,
                            trim_min,
                            trim_max,
                            trim_tolerance=0.4998):
        """
        Process predictions to create scaled, trimmed
        and rounded predictions.

        Parameters
        ----------
        df_test_predictions : pd.DataFrame
            Data frame containing the test set predictions.
        train_predictions_mean : float
            The mean of the predictions on the training set.
        train_predictions_sd : float
            The std. dev. of the predictions on the training
            set.
        human_labels_mean : float
            The mean of the human scores used to train the
            model.
        human_labels_sd : float
            The std. dev. of the human scores used to train
            the model.
        trim_min : float
            The lowest score on the score point, used for
            trimming the raw regression predictions.
        trim_max : float
            The highest score on the score point, used for
            trimming the raw regression predictions.
        trim_tolerance: float
            Tolerance to be added to trim_max and substracted from
            trim_min. Defaults to 0.4998.

        Returns
        -------
        df_pred_processed : pd.DataFrame
            Data frame containing the various trimmed
            and rounded predictions.
        """

        # rescale the test set predictions by boosting
        # them to match the human mean and SD
        scaled_test_predictions = (df_test_predictions['raw'] -
                                   train_predictions_mean) / train_predictions_sd
        scaled_test_predictions = scaled_test_predictions * human_labels_sd + human_labels_mean

        df_pred_process = df_test_predictions.copy()
        df_pred_process['scale'] = scaled_test_predictions

        # trim and round the predictions before running the analyses
        df_pred_process['raw_trim'] = FeaturePreprocessor.trim(df_pred_process['raw'],
                                                               trim_min,
                                                               trim_max,
                                                               trim_tolerance)

        df_pred_process['raw_trim_round'] = np.rint(df_pred_process['raw_trim'])
        df_pred_process['raw_trim_round'] = df_pred_process['raw_trim_round'].astype('int64')

        df_pred_process['scale_trim'] = FeaturePreprocessor.trim(df_pred_process['scale'],
                                                                 trim_min,
                                                                 trim_max,
                                                                 trim_tolerance)

        df_pred_process['scale_trim_round'] = np.rint(df_pred_process['scale_trim'])
        df_pred_process['scale_trim_round'] = df_pred_process['scale_trim_round'].astype('int64')

        return df_pred_process

    def filter_on_flag_columns(self,
                               df,
                               flag_column_dict):
        """
        Check that all flag_columns are present in the given
        data frame, convert these columns to strings and filter
        out the values which do not match the condition in
        `flag_column_dict`.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to filter on.
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

        df = df.copy()

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

            flag_column_dict_to_float = {key: list(map(convert_to_float, value))
                                         for (key, value)
                                         in flag_column_dict.items()}

            # and now convert the the values in the feature column
            # in the data frame
            df_new = df[flag_columns].copy()
            df_new = df_new.applymap(convert_to_float)

            # identify responses with values which satisfy the condition
            full_mask = df_new.isin(flag_column_dict_to_float)
            flag_mask = full_mask[list(flag_column_dict_to_float.keys())].all(1)

            # return the columns from the original frame that was passed in
            # so that all data types remain the same and are not changed
            df_responses_with_requested_flags = df[flag_mask].copy()
            df_responses_with_excluded_flags = df[~flag_mask].copy()

            # make sure that the remaining data frame is not empty
            if len(df_responses_with_requested_flags) == 0:
                raise ValueError("No responses remaining after filtering "
                                 "on flag columns. No further analysis can "
                                 "be run.")
            # reset the index
            df_responses_with_requested_flags.reset_index(drop=True,
                                                          inplace=True)

            df_responses_with_excluded_flags.reset_index(drop=True,
                                                         inplace=True)

            return (df_responses_with_requested_flags,
                    df_responses_with_excluded_flags)

    def generate_feature_names(self,
                               df,
                               reserved_column_names,
                               feature_subset_specs,
                               feature_subset):
        """
        Generate the feature names from the column
        names of the given data frame and select the
        specified subset of features.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame from which to generate feature names.
        reserved_column_names : list
            Names of reserved columns.
        feature_subset_specs : pd.DataFrame
            Feature subset specs
        feature_subset : str
            Feature subset column.

        Returns
        -------
        feautre_names : list
            A list of features names.
        """

        df = df.copy()

        # Exclude the reserved names
        possible_feature_names = [cname for cname in df.columns
                                  if cname not in reserved_column_names]

        # Select the features by subset.
        # In the future, we may add option to select
        # by other methods, if needed.
        if feature_subset is not None:
            feature_names = FeatureSubsetProcessor.select_by_subset(possible_feature_names,
                                                                    feature_subset_specs,
                                                                    feature_subset)
        else:
            feature_names = possible_feature_names
        return feature_names

    def preprocess_feature(self,
                           values,
                           feature_name,
                           feature_transform,
                           feature_mean,
                           feature_sd,
                           exclude_zero_sd=False,
                           raise_error=True,
                           truncations=None):
        """
        Remove outliers and transform the values in the given numpy array
        using the given outlier and transformation parameters. The values
        are assumed for the given feature name.

        Parameters
        ----------
        values : np.array
            The feature values to preprocess
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
            std. dev.
            Defaults to False.
        raise_error : bool, optional
            Raise error if any of the transformations lead to inf values
            or may change the ranking of feature values.
            Defaults to True.
        truncations : pd.DataFrame or None, optional
            The truncations set, if we are using pre-defined
            truncation values. Otherwise, None.
            Defaults to None.

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

        if truncations is not None:

            # clamp outlier values using the truncations set
            features_no_outliers = self.remove_outliers_using_truncations(values,
                                                                          feature_name,
                                                                          truncations)

        else:

            # clamp any outlier values that are 4 standard deviations
            # away from the mean
            features_no_outliers = self.remove_outliers(values,
                                                        mean=feature_mean,
                                                        sd=feature_sd)

        # apply the requested transformation to the feature
        transformed_feature = FeatureTransformer.transform_feature(features_no_outliers,
                                                                   feature_name,
                                                                   feature_transform,
                                                                   raise_error=raise_error)

        # check the standard deviation of the transformed feature
        # we set ddof to 1 so that np.std gave the same result as pandas .std
        # we also set the tolerance limit to account for cases where std
        # is computed as a very low decimal rather than 0
        # We only do this for the training set.
        if exclude_zero_sd:
            feature_sd = np.std(transformed_feature, ddof=1)
            if np.isclose(feature_sd, 0, atol=1e-07):
                raise ValueError("The standard deviation for "
                                 "feature {} is 0 after pre-processing. "
                                 "Please exclude this feature and re-run "
                                 "the experiment.".format(feature_name))

        return transformed_feature

    def preprocess_features(self,
                            df_train,
                            df_test,
                            df_feature_specs,
                            standardize_features=True,
                            use_truncations=False):
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
        standardize_features : bool, optional
            Whether to standardize the features
            Defaults to True.
        use_truncations : bool, optional
            Whether we should use the truncation set
            for removing outliers.
            Defaults to False.

        Returns
        -------
        df_train_preprocessed : pd.DataFrame
            DataFrame with preprocessed training data
        df_test_preprocessed : pd.DataFrame
            DataFrame with preprocessed test data
        df_feature_info : pd.DataFrame
            DataFrame with feature information
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

        # if we are should be using truncations, then we create the truncations
        # set from the feature specifications
        if use_truncations:
            truncations = df_feature_specs[['feature', 'min', 'max']].set_index('feature')
        else:
            truncations = None

        # now iterate over each feature
        for feature_name in df_feature_specs['feature']:

            feature_transformation = df_feature_specs.at[feature_name, 'transform']
            feature_sign = df_feature_specs.at[feature_name, 'sign']

            train_feature_mean = df_train[feature_name].mean()
            train_feature_sd = df_train[feature_name].std()

            training_feature_values = df_train[feature_name].values
            df_train_preprocessed[feature_name] = self.preprocess_feature(training_feature_values,
                                                                          feature_name,
                                                                          feature_transformation,
                                                                          train_feature_mean,
                                                                          train_feature_sd,
                                                                          exclude_zero_sd=True,
                                                                          truncations=truncations)

            testing_feature_values = df_test[feature_name].values
            df_test_preprocessed[feature_name] = self.preprocess_feature(testing_feature_values,
                                                                         feature_name,
                                                                         feature_transformation,
                                                                         train_feature_mean,
                                                                         train_feature_sd,
                                                                         truncations=truncations)

            # Standardize the features using the mean and sd computed on the
            # training set. These are computed separately because we need to
            # get the mean of transformed feature before standardization.
            train_transformed_mean = df_train_preprocessed[feature_name].mean()
            train_transformed_sd = df_train_preprocessed[feature_name].std()

            if standardize_features:

                df_train_without_mean = (df_train_preprocessed[feature_name] -
                                         train_transformed_mean)
                df_train_preprocessed[feature_name] = df_train_without_mean / train_transformed_sd

                df_test_without_mean = (df_test_preprocessed[feature_name] -
                                        train_transformed_mean)
                df_test_preprocessed[feature_name] = df_test_without_mean / train_transformed_sd

            # Multiply both train and test feature by sign.
            df_train_preprocessed[feature_name] = (df_train_preprocessed[feature_name] *
                                                   feature_sign)
            df_test_preprocessed[feature_name] = (df_test_preprocessed[feature_name] *
                                                  feature_sign)

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

    def filter_data(self,
                    df,
                    label_column,
                    id_column,
                    length_column,
                    second_human_score_column,
                    candidate_column,
                    requested_feature_names,
                    reserved_column_names,
                    given_trim_min,
                    given_trim_max,
                    flag_column_dict,
                    subgroups,
                    exclude_zero_scores=True,
                    exclude_zero_sd=False,
                    feature_subset_specs=None,
                    feature_subset=None,
                    min_candidate_items=None,
                    use_fake_labels=False):
        """
        Filter the data to remove rows that have zero/non-numeric values
        for `label_column`. If feature_names are specified, check whether any
        features that are specifically requested in `feature_names`
        are missing from the data. If no feature_names are specified,
        these are generated based on column names and subset information,
        if available. The function then excludes non-numeric values for
        any feature. If the user requested to exclude candidates with less
        than min_items_per_candidates, such candidates are excluded.
        It also generates fake labels between 1 and 10 if
        `use_fake_parameters` is set to True. Finally, it renames the id
        and label column and splits the data into the data frame with
        feature values and score label, the data frame with information about
        subgroup and candidate (metadata) and the data frame with all other
        columns.

        Parameters
        ----------
        df : pd.DataFrame
            The DataFrame to filter.
        label_column : str
            The label column in the data.
        id_column : str
            The ID column in the data.
        length_column : str
            The length column in the data.
        second_human_score_column : str
            The second human score column in the data.
        candidate_column : str
            The candidate column in the data.
        requested_feature_names : list
            A list of requested feature names.
        reserved_column_names : list
            A list of reserved column names.
        given_trim_min : float
            The minimum trim value.
        given_trim_max : float
            The maximum trim value.
        flag_column_dict : dict
            A dictionary of flag columns.
        subgroups : list, optional
            A list of subgroups, if any.
        exclude_zero_scores : bool
            Whether to exclude zero scores.
            Defaults to True.
        exclude_zero_sd : bool, optional
            Whether to exclude zero standard deviation.
            Defaults to False.
        feature_subset_specs : pd.DataFrame, optional
            The feature_subset_specs DataFrame
            Defaults to None.
        feature_subset : str, optional
            The feature subset group (e.g. 'A').
            Defaults to None.
        min_candidate_items : int, optional
            The minimum number of items needed to include candidate.
            Defaults to None
        use_fake_labels : bool, optional
            Whether to use fake labels.
            Defaults to None.

        Returns
        -------
        df_filtered_features : pd.DataFrame
            DataFrame with filtered features
        df_filtered_metadata : pd.DataFrame
            DataFrame with filtered metadata
        df_filtered_other_columns : pd.DataFrame
            DataFrame with other columns filtered
        df_excluded : pd.DataFrame
            DataFrame with excluded records
        df_filtered_length : pd.DataFrame
            DataFrame with length column(s) filtered
        df_filtered_human_scores : pd.DataFrame
            DataFrame with human scores filtered
        df_responses_with_excluded_flags : pd.DataFrame
            A DataFrame containing responses with excluded flags
        trim_min : float
            The maximum trim value
        trim_max : float
            The minimum trim value
        feature_names  : list
            A list of feature names
        """

        # make sure that the columns specified in the
        # config file actually exist
        columns_to_check = [id_column, label_column]

        if length_column:
            columns_to_check.append(length_column)

        if second_human_score_column:
            columns_to_check.append(second_human_score_column)

        if candidate_column:
            columns_to_check.append(candidate_column)

        missing_columns = set(columns_to_check).difference(df.columns)
        if missing_columns:
            raise KeyError("Columns {} from the config file "
                           "do not exist in the data.".format(missing_columns))

        # it is possible for the `id_column` and `candidate_column` to be
        # set to the same column name in the CSV file, e.g., if there is
        # only one response per candidate. If this happens, we neeed to
        # create a duplicate column for candidates or id for the downstream
        # processing to work as usual.
        if id_column == candidate_column:
            # if the name for both columns is `candidate`, we need to
            # create a separate id_column name
            if id_column == 'candidate':
                df['spkitemid'] = df['candidate'].copy()
                id_column = 'spkitemid'
            # else we create a separate `candidate` column
            else:
                df['candidate'] = df[id_column].copy()
                candidate_column = 'candidate'

        df = self.rename_default_columns(df,
                                         requested_feature_names,
                                         id_column,
                                         label_column,
                                         second_human_score_column,
                                         length_column,
                                         None,
                                         candidate_column)

        # check that the id_column contains unique values
        if df['spkitemid'].size != df['spkitemid'].unique().size:
            raise ValueError("The data contains duplicate response IDs in "
                             "'{}'. Please make sure all response IDs are "
                             "unique and re-run the tool.".format(id_column))

        # Generate feature names if no specific features were requested by the user
        if len(requested_feature_names) == 0:
            feature_names = self.generate_feature_names(df,
                                                        reserved_column_names,
                                                        feature_subset_specs=feature_subset_specs,
                                                        feature_subset=feature_subset)
        else:
            feature_names = requested_feature_names

        # make sure that feature names do not contain reserved column names
        illegal_feature_names = set(feature_names).intersection(reserved_column_names)
        if illegal_feature_names:
            raise ValueError("The following reserved "
                             "column names cannot be "
                             "used as feature names: '{}'. "
                             "Please rename these columns "
                             "and re-run the "
                             "experiment.".format(', '.join(illegal_feature_names)))

        # check to make sure that the subgroup columns are all present
        df = FeaturePreprocessor.check_subgroups(df, subgroups)

        # filter out the responses based on flag columns
        (df_responses_with_requested_flags,
         df_responses_with_excluded_flags) = self.filter_on_flag_columns(df, flag_column_dict)

        # filter out the rows that have non-numeric or zero labels
        # unless we are going to generate fake labels in the first place
        if not use_fake_labels:
            (df_filtered,
             df_excluded) = self.filter_on_column(df_responses_with_requested_flags,
                                                  'sc1',
                                                  'spkitemid',
                                                  exclude_zeros=exclude_zero_scores)

            # make sure that the remaining data frame is not empty
            if len(df_filtered) == 0:
                raise ValueError("No responses remaining after filtering out "
                                 "non-numeric human scores. No further analysis "
                                 "can be run. ")

            trim_min = given_trim_min if given_trim_min else df_filtered['sc1'].min()
            trim_max = given_trim_max if given_trim_max else df_filtered['sc1'].max()
        else:
            df_filtered = df_responses_with_requested_flags.copy()
            trim_min = given_trim_min if given_trim_min else 1
            trim_max = given_trim_max if given_trim_max else 10
            logging.info("Generating labels randomly "
                         "from [{}, {}]".format(trim_min, trim_max))
            randgen = RandomState(seed=1234567890)
            df_filtered[label_column] = randgen.random_integers(trim_min,
                                                                trim_max,
                                                                size=len(df_filtered))

        # make sure there are no missing features in the data
        missing_features = set(feature_names).difference(df_filtered.columns)
        if not missing_features:
            # make sure all features selected for model building are numeric
            # and also replace any non-numeric feature values in already
            # excluded data with NaNs for consistency
            for feat in feature_names:
                df_excluded[feat] = pd.to_numeric(df_excluded[feat],
                                                  errors='coerce').astype(float)
                newdf, newdf_excluded = self.filter_on_column(df_filtered,
                                                              feat,
                                                              'spkitemid',
                                                              exclude_zeros=False,
                                                              exclude_zero_sd=exclude_zero_sd)
                del df_filtered
                df_filtered = newdf
                with np.errstate(divide='ignore'):
                    df_excluded = pd.concat([df_excluded, newdf_excluded], sort=True)

            # make sure that the remaining data frame is not empty
            if len(df_filtered) == 0:
                raise ValueError("No responses remaining after filtering "
                                 "out non-numeric feature values. No further "
                                 "analysis can be run.")

            # Raise warning if we excluded features that were
            # specified in the .json file because sd == 0.
            omitted_features = set(requested_feature_names).difference(df_filtered.columns)
            if omitted_features:
                raise ValueError("The following requested features "
                                 "were excluded because their standard "
                                 "deviation on the training set was 0: {}.\n"
                                 "Please edit the feature file to exclude "
                                 "these features and re-run the "
                                 "tool".format(', '.join(omitted_features)))
            # Update the feature names
            feature_names = [feature for feature in feature_names
                             if feature in df_filtered]
        else:
            raise KeyError("DataFrame does not contain "
                           "columns for all features specified in "
                           "the feature file. Please check for "
                           "capitalization and other spelling "
                           "errors and make sure the feature "
                           "names do not contain hyphens. "
                           "The data does not have columns "
                           "for the following features: "
                           "{}".format(', '.join(missing_features)))

        # if ``length_column`` exists, make sure it's converted to numeric;
        # values that cannot be coerced to numeric will be set to ``np.nan``
        if length_column:
            df_filtered['length'] = pd.to_numeric(df_filtered['length'], errors='coerce')

        # check the values for length column. We do this after filtering
        # to make sure we have removed responses that have not been
        # processed correctly. Else rename length column to
        # ##ORIGINAL_NAME##.
        if (length_column and
            (len(df_filtered[df_filtered['length'].isnull()]) != 0 or
                df_filtered['length'].std() <= 0)):
            logging.warning("The {} column either has missing values or a standard "
                            "deviation <= 0. No length-based analysis will be "
                            "provided. The column will be renamed as ##{}## and "
                            "saved in *train_other_columns.csv.".format(length_column,
                                                                        length_column))
            df_filtered.rename(columns={'length': '##{}##'.format(length_column)},
                               inplace=True)

        # if requested, exclude the candidates with less than X responses
        # left after filtering
        if min_candidate_items:
            (df_filtered_candidates,
             df_excluded_candidates) = FeaturePreprocessor.select_candidates(df_filtered,
                                                                             min_candidate_items)
            # check that there are still responses left for analysis
            if len(df_filtered_candidates) == 0:
                raise ValueError("After filtering non-numeric scores and "
                                 "non-numeric feature values there were "
                                 "no candidates with {} or more responses "
                                 "left for analysis".format(min_candidate_items))

            # redefine df_filtered
            df_filtered = df_filtered_candidates.copy()

            # update df_excluded
            df_excluded = pd.concat([df_excluded, df_excluded_candidates], sort=True)

        # create separate data frames for features and sc1, all other
        # information, and responses excluded during filtering
        not_other_columns = set()
        feature_columns = ['spkitemid', 'sc1'] + feature_names
        df_filtered_features = df_filtered[feature_columns]
        not_other_columns.update(feature_columns)

        metadata_columns = ['spkitemid'] + subgroups
        if candidate_column:
            metadata_columns.append('candidate')
        df_filtered_metadata = df_filtered[metadata_columns]
        not_other_columns.update(metadata_columns)

        df_filtered_length = pd.DataFrame()
        length_columns = ['spkitemid', 'length']
        if length_column and 'length' in df_filtered:
            df_filtered_length = df_filtered[length_columns]
            not_other_columns.update(length_columns)

        df_filtered_human_scores = pd.DataFrame()
        human_score_columns = ['spkitemid', 'sc1', 'sc2']
        if second_human_score_column and 'sc2' in df_filtered:
            df_filtered_human_scores = df_filtered[human_score_columns].copy()
            not_other_columns.update(['sc2'])

            # filter out any non-numeric value rows
            # as well as zeros, if we were asked to
            df_filtered_human_scores['sc2'] = pd.to_numeric(df_filtered_human_scores['sc2'],
                                                            errors='coerce').astype(float)
            if exclude_zero_scores:
                df_filtered_human_scores['sc2'] = df_filtered_human_scores['sc2'].replace(0,
                                                                                          np.nan)

        # we need to make sure that `spkitemid` is the first column
        df_excluded = df_excluded[['spkitemid'] + [column for column in df_excluded
                                                   if column != 'spkitemid']]

        # now extract all other columns and add 'spkitemid'
        other_columns = ['spkitemid'] + [column for column in df_filtered
                                         if column not in not_other_columns]
        df_filtered_other_columns = df_filtered[other_columns]

        return (df_filtered_features,
                df_filtered_metadata,
                df_filtered_other_columns,
                df_excluded,
                df_filtered_length,
                df_filtered_human_scores,
                df_responses_with_excluded_flags,
                trim_min,
                trim_max,
                feature_names)

    def process_data_rsmtool(self, config_obj, data_container_obj):
        """
        The main function that sets up the experiment by loading the
        training and evaluation data sets and preprocessing them. Raises
        appropriate exceptions .

        Parameters
        ----------
        config_obj : configuration_parser.Configuration
            A configuration object.
        data_container_obj : container.DataContainer
            A data container object.

        Returns
        -------
        config_obj : configuration_parser.Configuration
            A Configuration object.
        data_container : container.DataContainer
            A DataContainer object.

        Raises
        ------
        ValueError
            If the columns in the config file do not exist in the data.
        """
        train = data_container_obj.train
        test = data_container_obj.test
        feature_specs = data_container_obj.get_frame('feature_specs')
        feature_subset = data_container_obj.get_frame('feature_subset_specs')

        configdir = config_obj.configdir

        (test_file_location,
         train_file_location) = DataReader.locate_files([config_obj['test_file'],
                                                         config_obj['train_file']],
                                                        configdir)

        feature_subset_file = config_obj['feature_subset_file']

        if feature_subset_file is not None:
            feature_subset_file = DataReader.locate_files(feature_subset_file, configdir)

        # get the column name for the labels for the training and testing data
        train_label_column = config_obj['train_label_column']
        test_label_column = config_obj['test_label_column']

        # get the column name that will hold the ID for
        # both the training and the test data
        id_column = config_obj['id_column']

        # get the specified trim min, trim max and trim tolerance values
        (spec_trim_min,
         spec_trim_max,
         spec_trim_tolerance) = config_obj.get_trim_min_max_tolerance()

        # get the name of the optional column that
        # contains response length.
        length_column = config_obj['length_column']

        # get the name of the optional column that
        # contains the second human score
        second_human_score_column = config_obj['second_human_score_column']

        # get the name of the optional column that
        # contains the candidate ID
        candidate_column = config_obj['candidate_column']

        # if the test label column is the same as the
        # second human score column, raise an error
        if test_label_column == second_human_score_column:
            raise ValueError("'test_label_column' and "
                             "'second_human_score_column' cannot have the "
                             "same value.")

        # check if we are excluding candidates based on number of responses
        exclude_listwise = config_obj.check_exclude_listwise()
        min_items = config_obj['min_items_per_candidate']

        # get the name of the model that we want to train and
        # check that it's valid
        model_name = config_obj['model']
        model_type = self.check_model_name(model_name)

        # are we excluding zero scores?
        exclude_zero_scores = config_obj['exclude_zero_scores']

        # should we standardize the features
        standardize_features = config_obj['standardize_features']

        # if we are excluding zero scores but trim_min
        # is set to 0, then we need to warn the user
        if exclude_zero_scores and spec_trim_min == 0:
            logging.warning("'exclude_zero_scores' is set to True but "
                            "'trim_min' is set to 0. This may cause "
                            " unexpected behavior.")

        # are we filtering on any other columns?
        # is `flag_column` applied to training partition only
        # or both partitions?
        if 'flag_column_test' in config_obj:
            flag_partition = 'train'
        else:
            flag_partition = 'both'

        flag_column_dict = config_obj.check_flag_column(partition=flag_partition)
        flag_column_test_dict = config_obj.check_flag_column('flag_column_test',
                                                             partition='test')

        if (flag_column_dict and not flag_column_test_dict):
            flag_column_test_dict = flag_column_dict

        # are we generating fake labels?
        use_fake_train_labels = train_label_column == 'fake'
        use_fake_test_labels = test_label_column == 'fake'

        # are we using truncations from the feature specs?
        use_truncations = config_obj['use_truncation_thresholds']

        # get the subgroups if any
        subgroups = config_obj.get('subgroups')

        # are there specific general report sections we want to include?
        general_report_sections = config_obj['general_sections']

        # what about the special or custom sections?
        special_report_sections = config_obj['special_sections']

        custom_report_section_paths = config_obj['custom_sections']

        if custom_report_section_paths and configdir is not None:
            logging.info('Locating custom report sections')
            custom_report_sections = Reporter.locate_custom_sections(custom_report_section_paths,
                                                                     configdir)
        else:
            custom_report_sections = []

        section_order = config_obj['section_order']

        chosen_notebook_files = Reporter().get_ordered_notebook_files(general_report_sections,
                                                                      special_report_sections,
                                                                      custom_report_sections,
                                                                      section_order,
                                                                      subgroups,
                                                                      model_type=model_type,
                                                                      context='rsmtool')

        # Location of feature file
        feature_field = config_obj['features']

        feature_subset_field = config_obj['feature_subset']

        # if the user requested feature_subset file and feature subset,
        # read the file and check its format
        if feature_subset is not None and feature_subset_field:
            FeatureSubsetProcessor.check_feature_subset_file(feature_subset)

        # Do we need to automatically find the best transformations/change sign?
        select_transformations = config_obj['select_transformations']
        feature_sign = config_obj['sign']
        requested_features = []
        generate_feature_specs_automatically = True

        # if the feature field is a list, then simply
        # assign it to `requested_features`
        if isinstance(feature_field, list):
            requested_features = feature_field

        elif feature_field is not None:
            generate_feature_specs_automatically = False
            feature_specs = FeatureSpecsProcessor.validate_feature_specs(feature_specs,
                                                                         use_truncations)
            requested_features = feature_specs['feature'].tolist()

        # if we get to this point and both ``generate_feature_specs_automatically``
        # and ``use_truncations`` are True, then we need to raise an error
        if use_truncations and generate_feature_specs_automatically:
            raise ValueError('You have specified the ``use_truncations`` configuration '
                             'option, but a feature file could not be found.')

        # check to make sure that `length_column` or `second_human_score_column`
        # are not also included in the requested features, if they are specified
        if (length_column and
                length_column in requested_features):
            raise ValueError("The value of 'length_column' ('{}') cannot be "
                             "used as a model feature.".format(length_column))

        if (second_human_score_column and
                second_human_score_column in requested_features):
            raise ValueError("The value of 'second_human_score_column' ('{}') cannot be "
                             "used as a model feature.".format(second_human_score_column))

        # Specify column names that cannot be used as features
        reserved_column_names = list(set(['spkitemid', 'spkitemlab',
                                          'itemType', 'r1', 'r2', 'score',
                                          'sc', 'sc1', 'adj',
                                          train_label_column,
                                          test_label_column,
                                          id_column] + subgroups + list(flag_column_dict.keys())))

        # if `second_human_score_column` is specified, then
        # we need to add the original name as well as `sc2` to the list of reserved column
        # names. And same for 'length' and 'candidate', if `length_column`
        # and `candidate_column` are specified. We add both names to
        # simplify things downstream since neither the original name nor
        # the standardized name should be used as feature names
        if second_human_score_column:
            reserved_column_names.append(second_human_score_column)
            reserved_column_names.append('sc2')
        if length_column:
            reserved_column_names.append(length_column)
            reserved_column_names.append('length')
        if candidate_column:
            reserved_column_names.append(candidate_column)
            reserved_column_names.append('candidate')

        # remove duplicates (if any) from the list of reserved column names
        reserved_column_names = list(set(reserved_column_names))

        # Make sure that the training data as specified in the
        # config file actually exists on disk and if it does,
        # load it and filter out the bad rows and features with
        # zero standard deviation. Also double check that the requested
        # features exist in the data or obtain the feature names if
        # no feature file was given.
        (df_train_features,
         df_train_metadata,
         df_train_other_columns,
         df_train_excluded,
         df_train_length,
         _,
         df_train_flagged_responses,
         used_trim_min,
         used_trim_max,
         feature_names) = self.filter_data(train,
                                           train_label_column,
                                           id_column,
                                           length_column,
                                           None,
                                           candidate_column,
                                           requested_features,
                                           reserved_column_names,
                                           spec_trim_min,
                                           spec_trim_max,
                                           flag_column_dict,
                                           subgroups,
                                           exclude_zero_scores=exclude_zero_scores,
                                           exclude_zero_sd=True,
                                           feature_subset_specs=feature_subset,
                                           feature_subset=feature_subset_field,
                                           min_candidate_items=min_items,
                                           use_fake_labels=use_fake_train_labels)

        # Generate feature specifications now that we know what features to use
        if generate_feature_specs_automatically:
            if select_transformations:

                feature_specs = FeatureSpecsProcessor.generate_specs(df_train_features,
                                                                     feature_names,
                                                                     'sc1',
                                                                     feature_subset=feature_subset,
                                                                     feature_sign=feature_sign)
            else:
                feature_specs = FeatureSpecsProcessor.generate_default_specs(feature_names)

        # Do the same for the test data except we can ignore the trim min
        # and max since we already have that from the training data and
        # we have the feature_names when no feature file was specified.
        # We also allow features with 0 standard deviation in the test file.
        if (test_file_location == train_file_location and
                train_label_column == test_label_column):
            logging.warning('The same data file and label '
                            'column are used for both training '
                            'and evaluating the model. No second '
                            'score analysis will be performed, even '
                            'if requested.')

            df_test_features = df_train_features.copy()
            df_test_metadata = df_train_metadata.copy()
            df_test_excluded = df_train_excluded.copy()
            df_test_other_columns = df_train_other_columns.copy()
            df_test_flagged_responses = df_train_flagged_responses.copy()
            df_test_human_scores = pd.DataFrame()
        else:

            (df_test_features,
             df_test_metadata,
             df_test_other_columns,
             df_test_excluded,
             _,
             df_test_human_scores,
             df_test_flagged_responses,
             _, _, _) = self.filter_data(test,
                                         test_label_column,
                                         id_column,
                                         None,
                                         second_human_score_column,
                                         candidate_column,
                                         feature_names,
                                         reserved_column_names,
                                         used_trim_min,
                                         used_trim_max,
                                         flag_column_test_dict,
                                         subgroups,
                                         exclude_zero_scores=exclude_zero_scores,
                                         exclude_zero_sd=False,
                                         min_candidate_items=min_items,
                                         use_fake_labels=use_fake_test_labels)

        logging.info('Pre-processing training and test set features')
        (df_train_preprocessed_features,
         df_test_preprocessed_features,
         df_feature_info) = self.preprocess_features(df_train_features,
                                                     df_test_features,
                                                     feature_specs,
                                                     standardize_features,
                                                     use_truncations)

        # configuration options that either override previous values or are
        # entirely for internal use
        new_config_obj = config_obj.copy()
        internal_options_dict = {'chosen_notebook_files': chosen_notebook_files,
                                 'exclude_listwise': exclude_listwise,
                                 'feature_subset_file': feature_subset_file,
                                 'model_name': model_name,
                                 'model_type': model_type,
                                 'test_file_location': test_file_location,
                                 'train_file_location': train_file_location,
                                 'trim_min': used_trim_min,
                                 'trim_max': used_trim_max}

        for key, value in internal_options_dict.items():
            new_config_obj[key] = value

        new_container = [{'name': 'train_features',
                          'frame': df_train_features},
                         {'name': 'test_features',
                          'frame': df_test_features},
                         {'name': 'train_preprocessed_features',
                          'frame': df_train_preprocessed_features},
                         {'name': 'test_preprocessed_features',
                          'frame': df_test_preprocessed_features},
                         {'name': 'train_metadata', 'frame': df_train_metadata},
                         {'name': 'test_metadata', 'frame': df_test_metadata},
                         {'name': 'train_other_columns', 'frame': df_train_other_columns},
                         {'name': 'test_other_columns', 'frame': df_test_other_columns},
                         {'name': 'train_excluded', 'frame': df_train_excluded},
                         {'name': 'test_excluded', 'frame': df_test_excluded},
                         {'name': 'train_length', 'frame': df_train_length},
                         {'name': 'test_human_scores', 'frame': df_test_human_scores},
                         {'name': 'train_flagged', 'frame': df_train_flagged_responses},
                         {'name': 'test_flagged', 'frame': df_test_flagged_responses},
                         {'name': 'feature_specs', 'frame': feature_specs},
                         {'name': 'feature_info', 'frame': df_feature_info}]

        new_container = DataContainer(new_container)

        return new_config_obj, new_container

    def process_data_rsmeval(self, config_obj, data_container_obj):
        """
        The main function that sets up the experiment by loading the
        training and evaluation data sets and preprocessing them. Raises
        appropriate exceptions .

        Parameters
        ----------
        config_obj : configuration_parser.Configuration
            A configuration object.
        data_container_obj : container.DataContainer
            A data container object.

        Returns
        -------
        config_obj : configuration_parser.Configuration
            A new configuration object.
        data_congtainer : container.DataContainer
            A new data container object.

        Raises
        ------
        ValueError
        """

        # get the directory where the config file lives
        # if this is the 'expm' directory, then go
        # up one level.
        configpath = config_obj.configdir

        pred_file_location = DataReader.locate_files(config_obj['predictions_file'],
                                                     configpath)

        # get the column name for the labels for the training and testing data
        human_score_column = config_obj['human_score_column']
        system_score_column = config_obj['system_score_column']

        # if the human score column is the same as the
        # system score column, raise an error
        if human_score_column == system_score_column:
            raise ValueError("'human_score_column' and "
                             "'system_score_column' "
                             "cannot have the same value.")

        # get the name of the optional column that
        # contains the second human score
        second_human_score_column = config_obj['second_human_score_column']

        # if the human score column is the same as the
        # second human score column, raise an error
        if human_score_column == second_human_score_column:
            raise ValueError("'human_score_column' and "
                             "'second_human_score_column' "
                             "cannot have the same value.")

        # get the column name that will hold the ID for
        # both the training and the test data
        id_column = config_obj['id_column']

        # get the specified trim min and max, if any
        # and make sure they are numeric
        (spec_trim_min,
         spec_trim_max,
         spec_trim_tolerance) = config_obj.get_trim_min_max_tolerance()

        # get the subgroups if any
        subgroups = config_obj.get('subgroups')

        # get the candidate column if any and convert it to string
        candidate_column = config_obj['candidate_column']

        # check if we are excluding candidates based on number of responses
        exclude_listwise = config_obj.check_exclude_listwise()
        min_items_per_candidate = config_obj['min_items_per_candidate']

        general_report_sections = config_obj['general_sections']

        # get any special sections that the user might have specified
        special_report_sections = config_obj['special_sections']

        # get any custom sections and locate them to make sure
        # that they exist, otherwise raise an exception
        custom_report_section_paths = config_obj['custom_sections']
        if custom_report_section_paths:
            logging.info('Locating custom report sections')
            custom_report_sections = Reporter.locate_custom_sections(custom_report_section_paths,
                                                                     configpath)
        else:
            custom_report_sections = []

        section_order = config_obj['section_order']

        # check all sections values and order and get the
        # ordered list of notebook files
        chosen_notebook_files = Reporter().get_ordered_notebook_files(general_report_sections,
                                                                      special_report_sections,
                                                                      custom_report_sections,
                                                                      section_order,
                                                                      subgroups,
                                                                      model_type=None,
                                                                      context='rsmeval')

        # are we excluding zero scores?
        exclude_zero_scores = config_obj['exclude_zero_scores']

        # if we are excluding zero scores but trim_min
        # is set to 0, then we need to warn the user
        if exclude_zero_scores and spec_trim_min == 0:
            logging.warning("'exclude_zero_scores' is set to True but "
                            " 'trim_min' is set to 0. This may cause "
                            " unexpected behavior.")

        # are we filtering on any other columns?
        flag_column_dict = config_obj.check_flag_column(partition='test')

        # do we have the training set predictions and human scores CSV file
        scale_with = config_obj.get('scale_with')

        # use scaled predictions for the analyses unless
        # we were told not to
        use_scaled_predictions = (scale_with is not None)

        # log an appropriate message
        if scale_with is None:
            message = ('Assuming given system predictions '
                       'are unscaled and will be used as such.')
        elif scale_with == 'asis':
            message = ('Assuming given system predictions '
                       'are already scaled and will be used as such.')
        else:
            message = ('Assuming given system predictions '
                       'are unscaled and will be scaled before use.')

        logging.info(message)

        df_pred = data_container_obj.predictions

        # make sure that the columns specified in the config file actually exist

        # make sure that the columns specified in the config file actually exist
        columns_to_check = [id_column, human_score_column, system_score_column]

        if second_human_score_column:
            columns_to_check.append(second_human_score_column)

        if candidate_column:
            columns_to_check.append(candidate_column)

        missing_columns = set(columns_to_check).difference(df_pred.columns)
        if missing_columns:
            raise KeyError('Columns {} from the config file do not exist '
                           'in the predictions file.'.format(missing_columns))

        df_pred = self.rename_default_columns(df_pred,
                                              [],
                                              id_column,
                                              human_score_column,
                                              second_human_score_column,
                                              None,
                                              system_score_column,
                                              candidate_column)

        # check that the id_column contains unique values
        if df_pred['spkitemid'].size != df_pred['spkitemid'].unique().size:
            raise ValueError("The data contains duplicate response IDs "
                             "in '{}'. Please make sure all response IDs "
                             "are unique and re-run the tool.".format(id_column))

        df_pred = self.check_subgroups(df_pred, subgroups)

        # filter out the responses based on flag columns
        (df_responses_with_requested_flags,
         df_responses_with_excluded_flags) = self.filter_on_flag_columns(df_pred,
                                                                         flag_column_dict)

        # filter out rows that have non-numeric or zero human scores
        df_filtered, df_excluded = self.filter_on_column(df_responses_with_requested_flags,
                                                         'sc1',
                                                         'spkitemid',
                                                         exclude_zeros=exclude_zero_scores)

        # make sure that the remaining data frame is not empty
        if len(df_filtered) == 0:
            raise ValueError("No responses remaining after filtering out "
                             "non-numeric human scores. No further analysis "
                             "can be run. ")

        # Change all non-numeric machine scores in excluded
        # data to NaNs for consistency with rsmtool.
        # NOTE: This will *not* work if *all* of the values
        # in column are non-numeric. This is a known bug in
        # pandas: https://github.com/pydata/pandas/issues/9589
        # Therefore, we need add an additional check after this.
        df_excluded['raw'] = pd.to_numeric(df_excluded['raw'], errors='coerce').astype(float)

        # filter out the non-numeric machine scores from the rest of the data
        newdf, newdf_excluded = self.filter_on_column(df_filtered,
                                                      'raw',
                                                      'spkitemid',
                                                      exclude_zeros=False)

        del df_filtered
        df_filtered_pred = newdf

        # make sure that the remaining data frame is not empty
        if len(df_filtered_pred) == 0:
            raise ValueError("No responses remaining after filtering out "
                             "non-numeric machine scores. No further analysis "
                             "can be run. ")

        with np.errstate(divide='ignore'):
            df_excluded = pd.concat([df_excluded, newdf_excluded], sort=True)

        # if requested, exclude the candidates with less than X responses
        # left after filtering
        if exclude_listwise:
            (df_filtered_candidates,
             df_excluded_candidates) = self.select_candidates(df_filtered_pred,
                                                              min_items_per_candidate)

            # check that there are still responses left for analysis
            if len(df_filtered_candidates) == 0:
                raise ValueError("After filtering non-numeric human and system scores "
                                 "there were "
                                 "no candidates with {} or more responses "
                                 "left for analysis".format(str(min_items_per_candidate)))

            # redefine df_filtered_pred
            df_filtered_pred = df_filtered_candidates.copy()

            # update df_excluded
            df_excluded = pd.concat([df_excluded, df_excluded_candidates], sort=True)
            df_excluded = df_excluded[['spkitemid'] + [column for column in df_excluded
                                                       if column != 'spkitemid']]

        # set default values for scaling
        scale_pred_mean = 0
        scale_pred_sd = 1
        scale_human_mean = 0
        scale_human_sd = 1

        if data_container_obj.get_frame('scale') is not None:
            if ('sc1' not in data_container_obj.scale.columns and
                    'prediction' not in data_container_obj.scale.columns):
                raise KeyError('The CSV file specified for scaling ',
                               'must have the "prediction" and the "sc1" '
                               'columns.')
            else:
                scale_pred_mean, scale_pred_sd = (data_container_obj.scale['prediction'].mean(),
                                                  data_container_obj.scale['prediction'].std())
                scale_human_mean, scale_human_sd = (data_container_obj.scale['sc1'].mean(),
                                                    data_container_obj.scale['sc1'].std())

        logging.info('Processing predictions')
        df_pred_processed = self.process_predictions(df_filtered_pred,
                                                     scale_pred_mean,
                                                     scale_pred_sd,
                                                     scale_human_mean,
                                                     scale_human_sd,
                                                     spec_trim_min,
                                                     spec_trim_max,
                                                     spec_trim_tolerance)
        if not scale_with:
            expected_score_types = ['raw', 'raw_trim', 'raw_trim_round']
        elif scale_with == 'asis':
            expected_score_types = ['scale', 'scale_trim', 'scale_trim_round']
        else:
            expected_score_types = ['raw', 'raw_trim', 'raw_trim_round',
                                    'scale', 'scale_trim', 'scale_trim_round']

        # extract separated data frames that we will write out
        # as separate files
        not_other_columns = set()

        prediction_columns = ['spkitemid', 'sc1'] + expected_score_types
        df_predictions_only = df_pred_processed[prediction_columns]
        not_other_columns.update(prediction_columns)

        metadata_columns = ['spkitemid'] + subgroups
        if candidate_column:
            metadata_columns.append('candidate')
        df_test_metadata = df_filtered_pred[metadata_columns]
        not_other_columns.update(metadata_columns)

        df_test_human_scores = pd.DataFrame()
        human_score_columns = ['spkitemid', 'sc1', 'sc2']
        if second_human_score_column and 'sc2' in df_filtered_pred:
            df_test_human_scores = df_filtered_pred[human_score_columns].copy()
            not_other_columns.update(['sc2'])
            # filter out any non-numeric values nows
            # as well as zeros, if we were asked to
            df_test_human_scores['sc2'] = pd.to_numeric(df_test_human_scores['sc2'],
                                                        errors='coerce').astype(float)
            if exclude_zero_scores:
                df_test_human_scores['sc2'] = df_test_human_scores['sc2'].replace(0, np.nan)

        # remove 'spkitemid' from `not_other_columns`
        # because we want that in the other columns
        # data frame
        not_other_columns.remove('spkitemid')

        # extract all of the other columns in the predictions file
        other_columns = [column for column in df_filtered_pred.columns
                         if column not in not_other_columns]

        df_pred_other_columns = df_filtered_pred[other_columns]

        # add internal configuration options that we need
        new_config_obj = config_obj.copy()
        internal_options_dict = {'pred_file_location': pred_file_location,
                                 'exclude_listwise': exclude_listwise,
                                 'use_scaled_predictions': use_scaled_predictions,
                                 'chosen_notebook_files': chosen_notebook_files}

        for key, value in internal_options_dict.items():
            new_config_obj[key] = value

        # we need to make sure that `spkitemid` is the first column
        df_excluded = df_excluded[['spkitemid'] + [column for column in df_excluded
                                                   if column != 'spkitemid']]

        frames = [df_predictions_only,
                  df_test_metadata,
                  df_pred_other_columns,
                  df_test_human_scores,
                  df_excluded,
                  df_responses_with_excluded_flags]

        names = ['pred_test',
                 'test_metadata',
                 'test_other_columns',
                 'test_human_scores',
                 'test_excluded',
                 'test_responses_with_excluded_flags']

        new_container = [{'name': name, 'frame': frame}
                         for frame, name in zip(frames, names)]

        new_container = DataContainer(new_container)

        return new_config_obj, new_container

    def process_data_rsmpredict(self, config_obj, data_container_obj):
        """
        Process data for RSM predict.

        Parameters
        ----------
        config_obj : configuration_parser.Configuration
            A configuration object.
        data_container_obj : container.DataContainer
            A data container object.

        Returns
        -------
        config_obj : configuration_parser.Configuration
            A new configuration object.
        data_congtainer : container.DataContainer
            A new data container object.

        Raises
        ------
        KeyError
            If columns in the config file do not exist in the data
        ValueError
            If data contains duplicate response IDs
        """

        df_input = data_container_obj.input_features
        df_feature_info = data_container_obj.feature_info
        df_postproc_params = data_container_obj.postprocessing_params

        # get the column name that will hold the ID
        id_column = config_obj['id_column']

        # get the column name for human score (if any)
        human_score_column = config_obj['human_score_column']

        # get the column name for second human score (if any)
        second_human_score_column = config_obj['second_human_score_column']

        # get the column name for subgroups (if any)
        subgroups = config_obj['subgroups']

        # get the model
        model = config_obj['model']

        # should features be standardized?
        standardize_features = config_obj.get('standardize_features', True)

        # should we predict expected scores
        predict_expected_scores = config_obj['predict_expected_scores']

        # get the column names for flag columns (if any)
        flag_column_dict = config_obj.check_flag_column(partition='test')

        # get the name for the candidate_column (if any)
        candidate_column = config_obj['candidate_column']

        # make sure that the columns specified in the config file actually exist
        columns_to_check = [id_column] + subgroups + list(flag_column_dict.keys())

        # add subgroups and the flag columns to the list of columns
        # that will be added to the final file
        columns_to_copy = subgroups + list(flag_column_dict.keys())

        # human_score_column will be set to sc1 by default
        # we only raise an error if it's set to something else.
        # However, since we cannot distinguish whether the column was set
        # to sc1 by default or specified as such in the config file
        # we append it to output anyway as long as
        # it is in the input file
        if human_score_column != 'sc1' or 'sc1' in df_input.columns:
            columns_to_check.append(human_score_column)
            columns_to_copy.append('sc1')

        if candidate_column:
            columns_to_check.append(candidate_column)
            columns_to_copy.append('candidate')

        if second_human_score_column:
            columns_to_check.append(second_human_score_column)
            columns_to_copy.append('sc2')

        missing_columns = set(columns_to_check).difference(df_input.columns)
        if missing_columns:
            raise KeyError("Columns {} from the config file "
                           "do not exist in the data.".format(missing_columns))

        # rename all columns
        df_input = self.rename_default_columns(df_input,
                                               [],
                                               id_column,
                                               human_score_column,
                                               second_human_score_column,
                                               None,
                                               None,
                                               candidate_column=candidate_column)

        # check that the id_column contains unique values
        if df_input['spkitemid'].size != df_input['spkitemid'].unique().size:
            raise ValueError("The data contains repeated response IDs in {}. "
                             "Please make sure all response IDs are unique and "
                             "re-run the tool.".format(id_column))

        (df_features_preprocessed,
         df_excluded) = self.preprocess_new_data(df_input,
                                                 df_feature_info,
                                                 standardize_features)

        trim_min = df_postproc_params['trim_min'].values[0]
        trim_max = df_postproc_params['trim_max'].values[0]
        h1_mean = df_postproc_params['h1_mean'].values[0]
        h1_sd = df_postproc_params['h1_sd'].values[0]

        # if we are using a newly trained model, use trim_tolerance from the
        # df_postproc_params. If not, set it to the default value and show
        # warning
        if 'trim_tolerance' in df_postproc_params:
            trim_tolerance = df_postproc_params['trim_tolerance'].values[0]
        else:
            trim_tolerance = 0.4998
            logging.warning("The tolerance for trimming scores will be assumed to be 0.4998, "
                            "the default value in previous versions of RSMTool. "
                            "We recommend re-training the model to ensure future "
                            "compatibility.")

        # now generate the predictions for the features using this model
        logged_str = 'Generating predictions'
        logged_str += ' (expected scores).' if predict_expected_scores else '.'
        logging.info(logged_str)

        # compute minimum and maximum score for expected predictions
        min_score = int(np.rint(trim_min - trim_tolerance))
        max_score = int(np.rint(trim_max + trim_tolerance))

        df_predictions = model.predict(df_features_preprocessed,
                                       min_score,
                                       max_score,
                                       predict_expected=predict_expected_scores)

        train_predictions_mean = df_postproc_params['train_predictions_mean'].values[0]
        train_predictions_sd = df_postproc_params['train_predictions_sd'].values[0]

        df_predictions = self.process_predictions(df_predictions,
                                                  train_predictions_mean,
                                                  train_predictions_sd,
                                                  h1_mean,
                                                  h1_sd,
                                                  trim_min, trim_max,
                                                  trim_tolerance)

        # add back the columns that we were requested to copy if any
        if len(columns_to_copy) > 0:
            df_predictions_with_metadata = pd.merge(df_predictions,
                                                    df_input[['spkitemid'] + columns_to_copy])
            assert(len(df_predictions) == len(df_predictions_with_metadata))
        else:
            df_predictions_with_metadata = df_predictions.copy()

        # we need to make sure that `spkitemid` is the first column
        df_excluded = df_excluded[['spkitemid'] + [column for column in df_excluded
                                                   if column != 'spkitemid']]

        datasets = [{'name': 'features_processed', 'frame': df_features_preprocessed},
                    {'name': 'excluded', 'frame': df_excluded},
                    {'name': 'predictions_with_metadata', 'frame': df_predictions_with_metadata},
                    {'name': 'predictions', 'frame': df_predictions}]

        return config_obj, DataContainer(datasets)

    def process_data(self, config_obj, data_container_obj, context='rsmtool'):
        """
        Process the date for a given context.

        Parameters
        ----------
        config_obj : configuration_parser.Configuration
            A configuration object.
        data_container_obj : container.DataContainer
            A data container object.
        context : {'rsmtool', 'rsmeval', 'rsmpredict'}
            The context of the tool.

        Returns
        -------
        config_obj : configuration_parser.Configuration
            A new configuration object.
        data_congtainer : container.DataContainer
            A new data container object.

        Raises
        ------
        ValueError
            If the the context is not in {'rsmtool', 'rsmeval', 'rsmpredict'}
        """
        if context == 'rsmtool':
            return self.process_data_rsmtool(config_obj, data_container_obj)
        elif context == 'rsmeval':
            return self.process_data_rsmeval(config_obj, data_container_obj)
        elif context == 'rsmpredict':
            return self.process_data_rsmpredict(config_obj, data_container_obj)
        else:
            raise ValueError("The `context` argument must be in the set: "
                             "{'rsmtool', 'rsmeval', 'rsmpredict'}. "
                             "You passed `{}`.".format(context))

    def preprocess_new_data(self,
                            df_input,
                            df_feature_info,
                            standardize_features=True):
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
            Data frame with preprocessing parameters stored in the following columns ::

                - `feature` : the name of the feature; should match the feature names
                   in `df_input`.
                - `sign` : `1` or `-1`.  Indicates whether the feature value needs to
                   be multiplied by -1.
                - `transform` : :ref:`transformation <json_transformation>` that needs
                   to be applied to this feature
                - `train_mean`, `train_sd` : mean and standard deviation for outlier
                   truncation.
                - `train_transformed_mean`,`train_transformed_sd` : mean and standard
                   deviation for computing `z`-scores.
        standardize_features : bool, optional
            Whether the features should be standardized prior to prediction.
            Defaults to True.

        Returns
        -------
        df_features_preprocessed : pd.DataFrame
            Data frame with processed feature values
        df_excluded: pd.DataFrame
            Data frame with responses excluded from further analysis
            due to non-numeric feature values in the original file
            or after applying transformations. The data frame always contains the
            original feature values.

        Raises
        ------
        KeyError
            if some of the features specified in `df_feature_info` are not present
            in `df_input`
        ValueError
            if all responses have at least one non-numeric feature value and therefore
            no score can be generated for any of the responses.
        """
        # get the list of required features

        required_features = df_feature_info.index.tolist()

        # ensure that all the features that are needed by the model
        # are present in the input file
        input_feature_columns = [c for c in df_input if c != 'spkitemid']
        missing_features = set(required_features).difference(input_feature_columns)
        if missing_features:
            raise KeyError('The input feature file is missing the '
                           'following features: {}'.format(missing_features))

        extra_features = set(input_feature_columns).difference(required_features + ['spkitemid'])
        if extra_features:
            logging.warning('The following extraneous features '
                            'will be ignored: {}'.format(extra_features))

        # keep the required features plus the id
        features_to_keep = ['spkitemid'] + required_features

        # check if actually have the human scores for this data and add
        # sc1 to preprocessed features for consistency with other tools
        has_human_scores = 'sc1' in df_input
        if has_human_scores:
            features_to_keep.append('sc1')

        df_features = df_input[features_to_keep]

        # preprocess the feature values
        logging.info('Pre-processing input features')

        # first we need to filter out NaNs and any other
        # weird features, the same way we did for rsmtool.
        df_filtered = df_features.copy()
        df_excluded = pd.DataFrame(columns=df_filtered.columns)

        for feature_name in required_features:
            newdf, newdf_excluded = self.filter_on_column(df_filtered,
                                                          feature_name,
                                                          'spkitemid',
                                                          exclude_zeros=False,
                                                          exclude_zero_sd=False)
            del df_filtered
            df_filtered = newdf
            with np.errstate(divide='ignore'):
                df_excluded = pd.concat([df_excluded, newdf_excluded], sort=True)

        # make sure that the remaining data frame is not empty
        if len(df_filtered) == 0:
            raise ValueError("There are no responses left after "
                             "filtering out non-numeric feature values. No analysis "
                             "will be run")

        df_features = df_filtered.copy()
        df_features_preprocess = df_features.copy()
        for feature_name in required_features:

            feature_values = df_features_preprocess[feature_name].values

            feature_transformation = df_feature_info.loc[feature_name]['transform']
            feature_sign = df_feature_info.loc[feature_name]['sign']

            train_feature_mean = df_feature_info.loc[feature_name]['train_mean']
            train_feature_sd = df_feature_info.loc[feature_name]['train_sd']

            train_transformed_mean = df_feature_info.loc[feature_name]['train_transformed_mean']
            train_transformed_sd = df_feature_info.loc[feature_name]['train_transformed_sd']

            # transform the feature values and remove outliers
            df_features_preprocess[feature_name] = self.preprocess_feature(feature_values,
                                                                           feature_name,
                                                                           feature_transformation,
                                                                           train_feature_mean,
                                                                           train_feature_sd,
                                                                           exclude_zero_sd=False,
                                                                           raise_error=False)

            # filter the feature values once again to remove possible NaN and inf values that
            # might have emerged when applying transformations.
            # We do not need to do that if no transformation was applied.
            if feature_transformation not in ['raw', 'org']:
                # check that there are indeed inf or Nan values
                if np.isnan(df_features_preprocess[feature_name]).any() or \
                   np.isinf(df_features_preprocess[feature_name]).any():
                    (newdf,
                     newdf_excluded) = self.filter_on_column(df_features_preprocess,
                                                             feature_name,
                                                             'spkitemid',
                                                             exclude_zeros=False,
                                                             exclude_zero_sd=False)
                    del df_features_preprocess
                    df_features_preprocess = newdf

                    # add the response(s) with missing values to the excluded responses
                    # but make sure we are adding the original values, not the
                    # preprocessed ones
                    missing_values = df_features['spkitemid'].isin(newdf_excluded['spkitemid'])

                    df_excluded_original = df_features[missing_values].copy()
                    df_excluded = pd.merge(df_excluded, df_excluded_original, how='outer')

            # print(standardized_features)
            if standardize_features:

                # now standardize the feature values
                df_feature_minus_mean = (df_features_preprocess[feature_name] -
                                         train_transformed_mean)
                df_features_preprocess[feature_name] = (df_feature_minus_mean /
                                                        train_transformed_sd)

            # Multiply features by sign.
            df_features_preprocess[feature_name] = (df_features_preprocess[feature_name] *
                                                    feature_sign)

        # we need to make sure that `spkitemid` is the first column
        df_excluded = df_excluded[['spkitemid'] + [column for column in df_excluded
                                                   if column != 'spkitemid']]

        return (df_features_preprocess, df_excluded)
