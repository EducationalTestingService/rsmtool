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
    """Encapsulate feature transformation methods."""

    def __init__(self, logger=None):
        """Initialize the FeatureTransformer object."""
        self.logger = logger if logger else logging.getLogger(__name__)

    def apply_sqrt_transform(self, name, values, raise_error=True):
        """
        Apply the "sqrt" transform to ``values``.

        Parameters
        ----------
        name : str
            Name of the feature to transform.
        values : np.array
            Numpy array containing the feature values.
        raise_error : bool, optional
            If ``True``, raises an error if the transform is applied to
            a feature that has negative values.
            Defaults to ``True``.

        Returns
        -------
        new_data : np.array
            Numpy array containing the transformed feature values.

        Raises
        ------
        ValueError
            If the transform is applied to a feature that has negative
            values and ``raise_error`` is ``True``.
        """
        # check if the feature has any negative values
        if np.any(values < 0):
            if raise_error:
                raise ValueError(
                    f"The sqrt transformation should not be applied to "
                    f"feature {name} which can have negative values."
                )
            else:
                self.logger.warning(
                    f"The sqrt transformation was applied to feature "
                    f"{name} which has negative values for some responses. "
                    f"No system score will be generated for such responses"
                )

        with np.errstate(invalid="ignore"):
            new_data = np.sqrt(values)
        return new_data

    def apply_log_transform(self, name, values, raise_error=True):
        """
        Apply the "log" transform to ``values``.

        Parameters
        ----------
        name : str
            Name of the feature to transform.
        values : np.array
            Numpy array containing the feature values.
        raise_error : bool, optional
            If ``True``, raises an error if the transform is applied to
            a feature that has zero or negative values.
            Defaults to ``True``.

        Returns
        -------
        new_data : numpy array
            Numpy array containing the transformed feature
            values.

        Raises
        ------
        ValueError
            If the transform is applied to a feature that has
            zero or negative values and ``raise_error`` is ``True``.
        """
        # check if the feature has any zeros
        if np.any(values == 0):
            if raise_error:
                raise ValueError(
                    f"The log transformation should not be applied to "
                    f"feature {name} which can have a value of 0."
                )
            else:
                self.logger.warning(
                    f"The log transformation was applied to feature "
                    f"{name} which has a value of 0 for some responses. "
                    f"No system score will be generated for such responses."
                )

        # check if the feature has any negative values
        if np.any(values < 0):
            if raise_error:
                raise ValueError(
                    f"The log transformation should not be applied to "
                    f"feature {name} which can have negative values."
                )
            else:
                self.logger.warning(
                    f"The log transformation was applied to feature "
                    f"{name} which has negative values for some responses. "
                    f"No system score will be generated for such responses"
                )

        new_data = np.log(values)
        return new_data

    def apply_inverse_transform(self, name, values, raise_error=True, sd_multiplier=4):
        """
        Apply the "inv" (inverse) transform to ``values``.

        Parameters
        ----------
        name : str
            Name of the feature to transform.
        values : np.array
            Numpy array containing the feature values.
        raise_error : bool, optional
            If ``True``, raises an error if the transform is applied to
            a feature that has zero values or to a feature that has
           both positive and negative values.
            Defaults to ``True``.
        sd_multiplier : int, optional
            Use this std. dev. multiplier to compute the ceiling
            and floor for outlier removal and check that these
            are not equal to zero.
            Defaults to 4.

        Returns
        -------
        new_data : np.array
            Numpy array containing the transformed feature
            values.

        Raises
        ------
        ValueError
            If the transform is applied to a feature that is
            zero or to a feature that can have different
            signs, and ``raise_error`` is ``True``.
        """
        if np.any(values == 0):
            if raise_error:
                raise ValueError(
                    f"The inverse transformation should not be applied "
                    f"to feature {name} which can have a value of 0."
                )
            else:
                self.logger.warning(
                    f"The inverse transformation was applied to feature "
                    f"{name} which has a value of 0 for some responses. "
                    f"No system score will be generated for such responses."
                )

        # check if the floor or ceiling are zero
        data_mean = np.mean(values)
        data_sd = np.std(values, ddof=1)
        floor = data_mean - sd_multiplier * data_sd
        ceiling = data_mean + sd_multiplier * data_sd
        if floor == 0 or ceiling == 0:
            self.logger.warning(
                f"The floor/ceiling for feature {name} is zero after "
                f"applying the inverse transformation."
            )

        # check if the feature can be both positive and negative
        all_positive = np.all(np.abs(values) == values)
        all_negative = np.all(np.abs(values) == -values)
        if not (all_positive or all_negative):
            if raise_error:
                raise ValueError(
                    f"The inverse transformation should not be applied "
                    f"to feature {name} where the values can have different signs"
                )
            else:
                self.logger.warning(
                    f"The inverse transformation was applied to feature "
                    f"{name} where the values can have different signs. "
                    f"This can change the ranking of the responses."
                )

        with np.errstate(divide="ignore"):
            new_data = 1 / values

        return new_data

    def apply_add_one_inverse_transform(self, name, values, raise_error=True):
        """
        Apply the "addOneInv" (add one and invert) transform to ``values``.

        Parameters
        ----------
        name : str
            Name of the feature to transform.
        values : np.array
            Numpy array containing the feature values.
        raise_error : bool, optional
            If ``True``, raises an error if the transform is applied to
            a feature that has zero or negative values.
            Defaults to ``True``.

        Returns
        -------
        new_data : np.array
            Numpy array containing the transformed feature values.

        Raises
        ------
        ValueError
            If the transform is applied to a feature that
            has negative values and ``raise_error`` is ``True``.
        """
        # check if the feature has any negative values
        if np.any(values < 0):
            if raise_error:
                raise ValueError(
                    f"The addOneInv transformation should not be applied "
                    f"to feature {name} which can have negative values."
                )
            else:
                self.logger.warning(
                    f"The addOneInv transformation was applied to "
                    f"feature {name} which has negative values for "
                    f"some responses. This can change the ranking of "
                    f"the responses."
                )

        new_data = 1 / (values + 1)
        return new_data

    def apply_add_one_log_transform(self, name, values, raise_error=True):
        """
        Apply the "addOneLn" (add one and log) transform to ``values``.

        Parameters
        ----------
        name : str
            Name of the feature to transform.
        values : np.array
            Numpy array containing the feature values.
        raise_error : bool, optional
            If ``True``, raises an error if the transform is applied to
            a feature that has zero or negative values.
            Defaults to ``True``.

        Returns
        -------
        new_data : np.array
            Numpy array that contains the transformed feature values.

        Raises
        ------
        ValueError
            If the transform is applied to a feature that
            has negative values and ``raise_error`` is ``True``.
        """
        # check if the feature has any negative values
        if np.any(values < 0):
            if raise_error:
                raise ValueError(
                    f"The addOneLn transformation should not be applied "
                    f"to feature {name} which can have negative values."
                )
            else:
                self.logger.warning(
                    f"The log transformation was applied to feature "
                    f"{name} which has negative values for some responses. "
                    f"If the feature value remains negative after adding one, "
                    f"no score will be generated for such responses."
                )

        new_data = np.log(values + 1)
        return new_data

    def transform_feature(self, values, column_name, transform, raise_error=True):
        """
        Apply given transform to all values in the given numpy array.

        The values are assumed to be for the feature with the given name.

        Parameters
        ----------
        values : numpy array
            Numpy array containing the feature values.
        column_name : str
            Name of the feature to transform.
        transform : str
            Name of the transform to apply. One of {"inv", "sqrt", "log",
            "addOneInv", "addOneLn", "raw", "org"}.
        raise_error : bool, optional
            If ``True``, raise a ValueError if a transformation leads to
            invalid values or may change the ranking of the responses.
            Defaults to ``True``.

        Returns
        -------
        new_data : np.array
            Numpy array containing the transformed feature values.

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
        transforms = {
            "inv": self.apply_inverse_transform,
            "sqrt": self.apply_sqrt_transform,
            "log": self.apply_log_transform,
            "addOneInv": self.apply_add_one_inverse_transform,
            "addOneLn": self.apply_add_one_log_transform,
            "raw": lambda column_name, data, raise_error: data,
            "org": lambda column_name, data, raise_error: data,
        }

        # make sure we have a valid transform function
        if transform is None or transform not in transforms:
            raise ValueError(f"Unrecognized feature transformation:  {transform}")

        transformer = transforms.get(transform)
        new_data = transformer(column_name, values, raise_error)
        return new_data

    def find_feature_transform(self, feature_name, feature_value, scores):
        """
        Identify best transformation for feature given correlation with score.

        The best transformation is chosen based on the absolute Pearson
        correlation with human score.

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
            applicable_transformations = ["org", "inv"]
        else:
            applicable_transformations = ["org", "sqrt", "addOneInv", "addOneLn"]

        correlations = []
        for trans in applicable_transformations:
            try:
                transformed_value = self.transform_feature(feature_value, feature_name, trans)

                correlations.append(abs(pearsonr(transformed_value, scores)[0]))
            except ValueError:
                # If the transformation returns an error, append 0.
                correlations.append(0)
        best = np.argmax(correlations)
        best_transformation = applicable_transformations[best]
        return best_transformation
