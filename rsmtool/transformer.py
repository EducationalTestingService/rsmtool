"""
Class for transforming features.

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

import logging
import numpy as np

from scipy.stats.stats import pearsonr


class FeatureTransformer:
    """
    Encapsulate feature transformation methods.
    """

    @classmethod
    def apply_sqrt_transform(cls,
                             name,
                             values,
                             raise_error=True):
        """
        Apply the `sqrt` transform to `values`.

        Parameters
        ----------
        name : str
            Name of the feature to transform.
        values : numpy array
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

        if np.any(values < 0):
            if raise_error:
                raise ValueError("The sqrt transformation should not be "
                                 "applied to feature {} which can have "
                                 "negative values".format(name))
            else:
                logging.warning("The sqrt transformation was "
                                "applied to feature {} which has "
                                "negative values for some responses. "
                                "No system score will be generated "
                                "for such responses".format(name))

        with np.errstate(invalid='ignore'):
            new_data = np.sqrt(values)
        return new_data

    @classmethod
    def apply_log_transform(cls,
                            name,
                            values,
                            raise_error=True):
        """
        Apply the `log` transform to `values`.

        Parameters
        ----------
        name : str
            Name of the feature to transform.
        values : numpy array
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
        if np.any(values == 0):
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
        if np.any(values < 0):
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

        new_data = np.log(values)
        return new_data

    @classmethod
    def apply_inverse_transform(cls,
                                name,
                                values,
                                raise_error=True,
                                sd_multiplier=4):
        """
        Apply the inverse transform to `values`.

        Parameters
        ----------
        name : str
            Name of the feature to transform.
        values : numpy array
            Numpy array containing the feature values.
        raise_error : bool, optional
            When set to true, raises an error if the transform is applied to
            a feature that can be zero or to a feature that can have
            different signs.
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
        if np.any(values == 0):
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
        data_mean = np.mean(values)
        data_sd = np.std(values, ddof=1)
        floor = data_mean - sd_multiplier * data_sd
        ceiling = data_mean + sd_multiplier * data_sd
        if floor == 0 or ceiling == 0:
            logging.warning("The floor/ceiling for feature {} "
                            "is zero after applying the inverse "
                            "transformation".format(name))

        # check if the feature can be both positive and negative
        all_positive = np.all(np.abs(values) == values)
        all_negative = np.all(np.abs(values) == -values)
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
            new_data = 1 / values

        return new_data

    @classmethod
    def apply_add_one_inverse_transform(cls,
                                        name,
                                        values,
                                        raise_error=True):
        """
        Apply the add one and invert transform to `values`.

        Parameters
        ----------
        name : str
            Name of the feature to transform.
        values : np.array
            Numpy array containing the feature values.
        raise_error : bool, optional
            When set to true, raises an error if the transform is applied to
            a feature that has zero or negative values.

        Returns
        -------
        new_data : np.array
            Numpy array containing the transformed feature
            values.

        Raises
        ------
        ValueError
            If the transform is applied to a feature
            that can be negative and `raise_error` is set to True.
        """

        # check if the feature has any negative values
        if np.any(values < 0):
            if raise_error:
                raise ValueError("The addOneInv transformation should not "
                                 "be applied to feature {} which can have "
                                 "negative values".format(name))
            else:
                logging.warning("The addOneInv transformation was "
                                "applied to feature {} which has "
                                "negative values for some responses. "
                                "This can change the ranking of the "
                                "responses".format(name))

        new_data = 1 / (values + 1)
        return new_data

    @classmethod
    def apply_add_one_log_transform(cls,
                                    name,
                                    values,
                                    raise_error=True):
        """
        Apply the add one and log transform to `values`.

        Parameters
        ----------
        name : str
            Name of the feature to transform.
        values : numpy array
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
        if np.any(values < 0):
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

        new_data = np.log(values + 1)
        return new_data

    @classmethod
    def transform_feature(cls,
                          values,
                          column_name,
                          transform,
                          raise_error=True):
        """
        Applies the given transform to all of the values in the given
        numpy array. The values are assumed to be for the feature with
        the given name.

        Parameters
        ----------
        values : numpy array
            Numpy array containing the feature values.
        column_name : str
            Name of the feature to transform.
        transform : str
            Name of the transform to apply.
            Valid options include ::

                {'inv', 'sqrt', 'log',
                 'addOneInv', 'addOneLn',
                 'raw', 'org'}

        raise_error : bool, optional
            Raise a ValueError if a transformation leads to `Inf` values or may
            change the ranking of the responses

        Returns
        -------
        new_data : np.array
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

        transforms = {'inv': FeatureTransformer.apply_inverse_transform,
                      'sqrt': FeatureTransformer.apply_sqrt_transform,
                      'log': FeatureTransformer.apply_log_transform,
                      'addOneInv': FeatureTransformer.apply_add_one_inverse_transform,
                      'addOneLn': FeatureTransformer.apply_add_one_log_transform,
                      'raw': lambda column_name, data, raise_error: data,
                      'org': lambda column_name, data, raise_error: data}

        # make sure we have a valid transform function
        if transform is None or transform not in transforms:
            raise ValueError('Unrecognized feature transformation: '
                             ' {}'.format(transform))

        transformer = transforms.get(transform)
        new_data = transformer(column_name, values, raise_error)
        return new_data

    @classmethod
    def find_feature_transform(cls,
                               feature_name,
                               feature_value,
                               scores):
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
            applicable_transformations = ['org',
                                          'sqrt',
                                          'addOneInv',
                                          'addOneLn']

        correlations = []
        for trans in applicable_transformations:
            try:
                transformed_value = FeatureTransformer.transform_feature(feature_value,
                                                                         feature_name,
                                                                         trans)

                correlations.append(abs(pearsonr(transformed_value, scores)[0]))
            except ValueError:
                # If the transformation returns an error, append 0.
                correlations.append(0)
        best = np.argmax(correlations)
        best_transformation = applicable_transformations[best]
        return best_transformation
