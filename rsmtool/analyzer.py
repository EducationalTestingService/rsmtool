"""
Classes for analyzing RSMTool predictions, metrics, etc.

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

import numpy as np
import pandas as pd
import warnings

from functools import partial

from scipy.stats import kurtosis, pearsonr
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from skll.metrics import kappa
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from .container import DataContainer
from .utils.metrics import (agreement,
                            difference_of_standardized_means,
                            partial_correlations,
                            quadratic_weighted_kappa,
                            standardized_mean_difference)
from .utils.prmse import get_true_score_evaluations


class Analyzer:
    """
    Analyzer class, which performs analysis on all metrics,
    predictions, etc.
    """

    @staticmethod
    def check_frame_names(data_container,
                          dataframe_names):
        """
        Check to make sure all specified DataFrames
        are in the data container object.

        Parameters
        ----------
        data_container : container.DataContainer
            A DataContainer object
        dataframe_names : list of str
            The names of the DataFrames expected in the
            DataContainer object.

        Raises
        ------
        KeyError
            If a given dataframe_name is not in the DataContainer object.
        """
        for dataframe_name in dataframe_names:
            if dataframe_name not in data_container:
                raise KeyError('The DataFrame `{}` does not exist in the '
                               'DataContainer object.'.format(dataframe_name))

    @staticmethod
    def check_param_names(configuration_obj,
                          parameter_names):
        """
        Check to make sure all specified parameters
        are in the configuration object.

        Parameters
        ----------
        configuration_obj : configuration_parser.Configuration
            A configuration object
        parameter_names : list of str
            The names of the parameters (keys) expected in the
            Configuration object.

        Raises
        ------
        KeyError
            If a given parameter_name is not in the Configuration object.
        """

        for parameter_name in parameter_names:
            if parameter_name not in configuration_obj:
                raise KeyError('The parameter `{}` does not exist in the '
                               'Configuration object.'.format(parameter_name))

    @staticmethod
    def analyze_excluded_responses(df,
                                   features,
                                   header,
                                   exclude_zero_scores=True,
                                   exclude_listwise=False):
        """
        Compute statistics on the responses that were excluded from
        analyses, either in the training set or in the test set.

        Parameters
        ----------
        df : pandas DataFrame
            Data frame containing the excluded responses
        features : list of str
            List of column names containing the features
            to which we want to restrict the analyses.
        header : str
            String to be used as the table header for the
            output data frame.
        exclude_zero_scores : bool, optional
            Whether or not the zero-score responses
            should be counted in the exclusion statistics,
            defaults to True.
        exclude_listwise : bool, optional
            Whether or not the candidates were excluded
            based on minimal number of responses

        Returns
        -------
        df_full_crosstab : pandas DataFrame
            Two-dimensional data frame containing the
            exclusion statistics.
        """

        # create an empty output data frame
        df_full_crosstab = pd.DataFrame({'all features numeric': [0, 0, 0],
                                         'non-numeric feature values': [0, 0, 0]},
                                        index=['numeric non-zero human score',
                                               'zero human score',
                                               'non-numeric human score'])

        if not df.empty:
            # re-code human scores into numeric, missing or zero
            df['score_category'] = 'numeric non-zero human score'
            df.loc[df['sc1'].isnull(), 'score_category'] = 'non-numeric human score'
            df.loc[df['sc1'].astype(float) == 0, 'score_category'] = 'zero human score'

            # recode feature values: a response with at least one
            # missing feature is assigned 'non-numeric feature values'
            df_features_only = df[features + ['spkitemid']]
            null_feature_rows = df_features_only.isnull().any(axis=1)
            df_null_features = df_features_only[null_feature_rows]
            df['feat_category'] = 'all features numeric'
            df.loc[df['spkitemid'].isin(df_null_features['spkitemid']),
                   'feat_category'] = 'non-numeric feature values'

            # crosstabulate
            df_crosstab = pd.crosstab(df['score_category'],
                                      df['feat_category'])
            df_full_crosstab.update(df_crosstab)
            # convert back to integers as these are all counts
            df_full_crosstab = df_full_crosstab.astype(int)
            df_full_crosstab.insert(0, header, df_full_crosstab.index)

        if not exclude_listwise:
            # if we are not excluding listwise, rename the first cell so
            # that it is not set to zero
            assert(df_full_crosstab.loc['numeric non-zero human score',
                                        'all features numeric'] == 0)
            df_full_crosstab.loc['numeric non-zero human score',
                                 'all features numeric'] = '-'

            # if we are not excluding the zeros, rename the corresponding cells
            # so that they are not set to zero. We do not do this for listwise exclusion
            if not exclude_zero_scores:
                assert(df_full_crosstab.loc['zero human score',
                                            'all features numeric'] == 0)
                df_full_crosstab.loc['zero human score',
                                     'all features numeric'] = '-'

        return df_full_crosstab

    @staticmethod
    def analyze_used_responses(df_train, df_test, subgroups, candidate_column):
        """
        Compute statistics on the responses that were used in
        analyses, either in the training set or in the test set.

        Parameters
        ----------
        df_train : pandas DataFrame
            Data frame containing the response information
            for the training set.
        df_test : pandas DataFrame
            Data frame containing the response information
            for the test set.
        subgroups : list of str
            List of column names that contain grouping
            information.
        candidate_column : str
            Column name that contains candidate
            identification information.

        Returns
        -------
        df_analysis : pandas DataFrame
            Data frame containing information about the used
            responses.
        """

        # create a basic data frame for responses only
        train_responses = set(df_train['spkitemid'])
        test_responses = set(df_test['spkitemid'])

        rows = [{'partition': 'Training', 'responses': len(train_responses)},
                {'partition': 'Evaluation', 'responses': len(test_responses)},
                {'partition': 'Overlapping', 'responses': len(train_responses & test_responses)},
                {'partition': 'Total', 'responses': len(train_responses | test_responses)}]

        df_analysis = pd.DataFrame.from_dict(rows)

        columns = ['partition', 'responses'] + subgroups

        if candidate_column:
            train_candidates = set(df_train['candidate'])
            test_candidates = set(df_test['candidate'])
            df_analysis['candidates'] = [len(train_candidates), len(test_candidates),
                                         len(train_candidates & test_candidates),
                                         len(train_candidates | test_candidates)]

            columns = ['partition', 'responses', 'candidates'] + subgroups

        for group in subgroups:
            train_group = set(df_train[group])
            test_group = set(df_test[group])
            df_analysis[group] = [len(train_group), len(test_group),
                                  len(train_group & test_group),
                                  len(train_group | test_group)]

        df_analysis = df_analysis[columns]
        return df_analysis

    @staticmethod
    def analyze_used_predictions(df_test, subgroups, candidate_column):
        """
        Compute statistics on the predictions that were used in
        analyses.

        Parameters
        ----------
        df_test : pandas DataFrame
            Data frame containing the test set predictions.
        subgroups : list of str
            List of column names that contain grouping
            information.
        candidate_column : str
            Column name that contains candidate
            identification information.

        Returns
        -------
        df_analysis : pandas DataFrame
            Data frame containing information about the used
            predictions.
        """

        rows = [{'partition': 'Evaluation', 'responses': df_test['spkitemid'].size}]

        df_analysis = pd.DataFrame.from_dict(rows)
        df_columns = ['partition', 'responses'] + subgroups

        if candidate_column:
            df_analysis['candidates'] = [df_test['candidate'].unique().size]
            df_columns = ['partition', 'responses', 'candidates'] + subgroups

        for group in subgroups:
            df_analysis[group] = [df_test[group].unique().size]

        df_analysis = df_analysis[df_columns]
        return df_analysis

    @staticmethod
    def compute_basic_descriptives(df, selected_features):
        """
        Compute basic descriptive statistics for the columns
        in the given data frame.

        Parameters
        ----------
        df : pandas DataFrame
            Input data frame containing the feature values.
        selected_features : list of str
            List of feature names for which to compute
            the descriptives.

        Returns
        -------
        df_desc : pandas DataFrame
            DataFrame containing the descriptives for
            each of the features.
        """

        # select only feature columns
        df_desc = df[selected_features]

        # get the H1 scores
        scores = df['sc1']

        # compute correlations and p-values separately for efficiency
        cor_series = df_desc.apply(lambda s: pearsonr(s, scores))
        cors = cor_series.apply(lambda t: t[0])
        pvalues = cor_series.apply(lambda t: t[1])

        # create a data frame with all the descriptives
        df_output = pd.DataFrame({'mean': df_desc.mean(),
                                  'min': df_desc.min(),
                                  'max': df_desc.max(),
                                  'std. dev.': df_desc.std(),
                                  'skewness': df_desc.skew(),
                                  'kurtosis': df_desc.apply(lambda s: kurtosis(s, fisher=False)),
                                  'Correlation': cors,
                                  'p': pvalues,
                                  'N': len(df_desc)})

        # reorder the columns to make it look better
        df_output = df_output[['mean', 'std. dev.', 'min', 'max', 'skewness',
                               'kurtosis', 'Correlation', 'p', 'N']]

        return df_output

    @staticmethod
    def compute_percentiles(df,
                            selected_features,
                            percentiles=None):
        """
        Compute percentiles and outlier descriptives for the columns
        in the given data frame.

        Parameters
        ----------
        df : pandas DataFrame
            Input data frame containing the feature values.
        selected_features : list of str
            List of feature names for which to compute the
            percentile descriptives.
        percentiles : list of ints, optional
            The percentiles to calculate. If `None`, use the percentiles
            {1, 5, 25, 50, 75, 95, 99}.
            Defaults to None.

        Returns
        -------
        df_output : pandas DataFrame
            Data frame containing the percentile information
            for each of the features.
        """

        # select only feature columns
        df_desc = df[selected_features]

        # compute the various percentile levels
        if percentiles is None:
            percentiles = [1, 5, 25, 50, 75, 95, 99]

        df_output = df_desc.apply(lambda series: pd.Series(np.percentile(series,
                                                                         percentiles,
                                                                         interpolation='lower')))
        df_output = df_output.transpose()

        # change the column names to be more readable
        df_output.columns = ['{}%'.format(p) for p in percentiles]

        # add the inter-quartile range column
        df_output['IQR'] = df_output['75%'] - df_output['25%']

        # compute the various outlier statistics
        mild_upper = df_output['75%'] + 1.5 * df_output['IQR']
        mild_bottom = df_output['25%'] - 1.5 * df_output['IQR']

        extreme_upper = df_output['75%'] + 3 * df_output['IQR']
        extreme_bottom = df_output['25%'] - 3 * df_output['IQR']

        # compute the mild and extreme outliers
        num_mild_outliers = {}
        num_extreme_outliers = {}
        for c in df_desc.columns:
            is_extreme = (df_desc[c] <= extreme_bottom[c]) | (df_desc[c] >= extreme_upper[c])

            is_mild = ((df_desc[c] > extreme_bottom[c]) & (df_desc[c] <= mild_bottom[c]))
            is_mild = is_mild | ((df_desc[c] >= mild_upper[c]) & (df_desc[c] < extreme_upper[c]))
            num_mild_outliers[c] = len(df_desc[is_mild])
            num_extreme_outliers[c] = len(df_desc[is_extreme])

        # add those to the output data frame
        df_output['Mild outliers'] = pd.Series(num_mild_outliers)
        df_output['Extreme outliers'] = pd.Series(num_extreme_outliers)

        return df_output

    @staticmethod
    def compute_outliers(df, selected_features):
        """
        Compute the number and percentage of outliers
        outside mean +/- 4 SD for the given columns
        in the given data frame.

        Parameters
        ----------
        df : pandas DataFrame
            Input data frame containing the feature values.
        selected_features : list of str
            List of feature names for which to compute
            outlier information.

        Returns
        -------
        df_output : pandas DataFrame
            Data frame containing outlier information
            for each of the features.
        """

        # select only feature columns
        df_desc = df[selected_features]

        # compute the means and standard deviations
        means = df_desc.mean()
        stds = df_desc.std()

        # compute the number of upper and lower outliers
        lower_outliers = {}
        upper_outliers = {}
        for c in df_desc.columns:
            lower_outliers[c] = len(df_desc[df_desc[c] < means[c] - 4 * stds[c]])
            upper_outliers[c] = len(df_desc[df_desc[c] > means[c] + 4 * stds[c]])

        # generate the output data frame
        lower_s = pd.Series(lower_outliers)
        upper_s = pd.Series(upper_outliers)
        both_s = lower_s + upper_s
        df_output = pd.DataFrame({'lower': lower_s,
                                  'upper': upper_s,
                                  'both': both_s,
                                  'lowerperc': round(lower_s / len(df_desc) * 100, 2),
                                  'upperperc': round(upper_s / len(df_desc) * 100, 2),
                                  'bothperc': round(both_s / len(df_desc) * 100, 2)})

        return df_output

    @staticmethod
    def compute_pca(df, selected_features):
        """
        Compute the PCA decomposition of features in the given data
        frame, restricted to the given columns.
        The number of components is set to be
        min(n_features, n_samples).

        Parameters
        ----------
        df : pandas DataFrame
            Input data frame containing feature values.
        selected_features : list of str
            List of feature names to be used in the
            PCA decomposition.

        Returns
        -------
        df_components : pandas DataFrame
            Data frame containing the PCA components.
        df_variance : pandas DataFrame
            Data frame containing the variance information.
        """

        # remove the spkitemid and sc1 column

        df_pca = df[selected_features]

        # fit the PCA
        n_components = min(len(selected_features), len(df_pca))
        pca = PCA(n_components=n_components)
        pca.fit(df_pca)

        df_components = pd.DataFrame(pca.components_)
        n_components = len(df_components)
        df_components.columns = selected_features
        df_components.index = ['PC{}'.format(i) for i in range(1, n_components + 1)]
        df_components = df_components.transpose()

        # compute the variance data frame
        df_variance = {'Eigenvalues': pca.explained_variance_,
                       'Percentage of variance': pca.explained_variance_ratio_,
                       'Cumulative percentage of '
                       'variance': np.cumsum(pca.explained_variance_ratio_)
                       }

        df_variance = pd.DataFrame(df_variance)

        # reorder the columns
        df_variance = df_variance[['Eigenvalues', 'Percentage of variance',
                                   'Cumulative percentage of variance']]

        # set the row names and take the transpose
        df_variance.index = ['PC{}'.format(i) for i in range(1, n_components + 1)]
        df_variance = df_variance.transpose()

        return df_components, df_variance

    @staticmethod
    def correlation_helper(df,
                           target_variable,
                           grouping_variable,
                           include_length=False):
        """
        A helper function to compute marginal and partial correlations of
        all the columns in the given data frame against the target variable
        separately for each level in the the grouping variable.
        If `include_length` is True, it additionally computes partial
        correlations of each column in the data frame against the target
        variable after controlling for the `length` column.

        Parameters
        ----------
        df : pandas DataFrame
            Input data frame containing numeric feature values, the numeric
            `target variable` and the `grouping variable`.
        target_variable: str
            The name of the column used as a reference for computing correlations.
        grouping_variable: str
            The name of the column defining groups in the data
        include_length: bool, optional
            If True compute additional partial correlations of each column
            in the data frame against `target variable` only partialling out
            `length` column.

        Returns
        -------
        df_target_cors : pandas DataFrame
            Data frame containing Pearson's correlation coefficients for
            marginal correlations between features and `target_variable`.
        df_target_partcors : pandas DataFrame
            Data frame containing Pearson's correlation coefficients for
            partial correlations between each feature and `target_variable`
            after controlling for all other features. If include_length is
            set to True, `length` will not be included into partial
            correlation computation.
        df_target_partcors_no_length: pandas DataFrame
            If `include_length` is set to `true`: Data frame containing
            Pearson's correlation coefficients for partial correlations
            between each feature and `target_variable` after controlling
            for `length`. Otherwise, it will be an empty data frame.
        """

        # group by the group columns
        grouped = df.groupby(grouping_variable)

        df_target_cors = pd.DataFrame()
        df_target_pcorr = pd.DataFrame()
        df_target_pcorr_no_length = pd.DataFrame()

        for group, df_group in grouped:
            df_group = df_group.drop(grouping_variable, 1)

            # first check if we have at least 2 cases and return np.nan otherwise
            if len(df_group) == 1:
                df_target_cors[group] = pd.Series(data=np.nan,
                                                  index=df_group.columns)
                df_target_pcorr[group] = pd.Series(data=np.nan,
                                                   index=df_group.columns)
                df_target_pcorr_no_length[group] = pd.Series(data=np.nan,
                                                             index=df_group.columns)
            else:
                # if we are asked to include length, that means 'length' is
                # in the data frame which means that we want to exclude that
                # before computing the regular marginal and partial correlations
                if not include_length:
                    df_target_cors[group] = df_group.apply(lambda s:
                                                           pearsonr(s,
                                                                    df_group[target_variable])[0])
                    df_target_pcorr[group] = partial_correlations(df_group)[target_variable]
                else:
                    df_group_no_length = df_group.drop('length', axis=1)

                    partial_pearsonr = partial(pearsonr, y=df_group_no_length[target_variable])
                    df_target_cors[group] = df_group_no_length.apply(lambda s:
                                                                     partial_pearsonr(s)[0])

                    df_target_pcorr[group] = partial_correlations(df_group_no_length)[target_variable]
                    pcor_dict = {}
                    columns = [c for c in df_group.columns if c not in ['sc1', 'length']]
                    for c in columns:
                        pcor_dict[c] = partial_correlations(df_group[[c,
                                                                      'sc1',
                                                                      'length']])['sc1'][c]
                    df_target_pcorr_no_length[group] = pd.Series(pcor_dict)

        # remove the row containing the correlation of the target variable
        # with itself and take the transpose
        df_target_cors = df_target_cors.drop(target_variable).transpose()
        df_target_pcorr = df_target_pcorr.drop(target_variable).transpose()
        df_target_pcorr_no_length = df_target_pcorr_no_length.transpose()

        return (df_target_cors,
                df_target_pcorr,
                df_target_pcorr_no_length)

    @staticmethod
    def metrics_helper(human_scores,
                       system_scores,
                       population_human_score_sd=None,
                       population_system_score_sd=None,
                       population_human_score_mn=None,
                       population_system_score_mn=None,
                       smd_method='unpooled',
                       use_diff_std_means=False):
        """
        This is a helper function that computes several basic
        association metrics between the system scores and the
        human scores.

        Parameters
        ----------
        human_scores : pandas Series
            Series containing numeric human (reference) scores.
        system_scores: pandas Series
            Series containing numeric scores predicted by the model.
        population_human_score_sd : float, optional
            Reference standard deviation for human scores. If `smd_method='williamson'`, this is
            used to compute SMD and should be the standard deviation for the whole population. If
            `use_diff_std_means=True`, this must be used with `population_human_score_mn`.
            Otherwise, it is ignored.
            Defaults to None.
        population_system_score_sd : float, optional
            Reference standard deviation for system scores. If `smd_method='williamson'`, this is
            used to compute SMD and should be the standard deviation for the whole population.If
            `use_diff_std_means=True`, this must be used with `population_system_score_mn`.
            Otherwise, it is ignored.
            Defaults to None.
        population_human_score_mn : float, optional
            Reference mean for human scores. If `use_diff_std_means=True`, this must be used with
            `population_human_score_sd`. Otherwise, it is ignored.
            Defaults to None.
        population_system_score_mn : float, optional
            Reference mean for system scores. If  `use_diff_std_means=True`, this must be used with
            `population_system_score_sd`. Otherwise, it is ignored.
            Defaults to None.
        smd_method : {'williamson', 'johnson', pooled', 'unpooled'}, optional
            The SMD method to use, only used if `use_diff_std_means=False`.
            All methods have the same numerator mean(`y_pred`) - mean(`y_true_observed`)
            and the following denominators :

            - `williamson`: pooled population standard deviation of `y_true_observed` and `y_pred`
            - `johnson`: population standard deviation of `y_true_observed`.
            - `pooled`: pooled standard deviation of `y_true_observed` and `y_pred` for this group.
            - `unpooled`: standard deviation of `y_true_observed` for this group.

            Defaults to 'unpooled'.

        use_diff_std_means : bool, optional
            Whether to use the difference of standardized means, rather than the standardized mean
            difference. This is most useful with subgroup analysis.
            Defaults to False.

        Returns
        -------
        metrics: pandas Series
            Series containing different evaluation metrics comparing human
            and system scores. The following metrics are included :

            - `kappa`:  unweighted Cohen's kappa
            - `wtkappa`:  quadratic weighted kappa
            - `exact_agr`: exact agreement
            - `adj_agr`: adjacent agreement with tolerance set to 1
            - One of the following :

              * `SMD`: standardized mean difference, if `use_diff_std_means=False`
              * `DSM`: difference of standardized means, if `use_diff_std_means=True`

            - `corr`: Pearson's r
            - `R2`: r squared
            - `RMSE`: root mean square error
            - `sys_min`: min system score
            - `sys_max`: max system score
            - `sys_mean`: mean system score (ddof=1)
            - `sys_sd`: standard deviation of system scores (ddof=1)
            - `h_min`: min human score
            - `h_max`: max human score
            - `h_mean`: mean human score (ddof=1)
            - `h_sd`: standard deviation of human scores (ddof=1)
            - `N`: total number of responses
        """

        # compute the kappas
        unweighted_kappa = kappa(human_scores, system_scores)
        weighted_kappa = quadratic_weighted_kappa(human_scores,
                                                  system_scores)

        # compute the agreement statistics
        human_system_agreement = agreement(human_scores, system_scores)
        human_system_adjacent_agreement = agreement(human_scores,
                                                    system_scores,
                                                    tolerance=1)

        # compute the Pearson correlation after removing
        # any cases where either of the scores are NaNs.
        df = pd.DataFrame({'human': human_scores,
                           'system': system_scores}).dropna(how='any')

        if (len(df) == 1 or
                len(df['human'].unique()) == 1 or
                len(df['system'].unique()) == 1):
            # set correlations to 1 if we have a single instance or zero variance
            correlations = np.nan
        else:
            correlations = pearsonr(df['human'], df['system'])[0]

        # compute the min/max/mean/std. dev. for the system and human scores
        min_system_score = np.min(system_scores)
        min_human_score = np.min(human_scores)

        max_system_score = np.max(system_scores)
        max_human_score = np.max(human_scores)

        mean_system_score = np.mean(system_scores)
        mean_human_score = np.mean(human_scores)

        system_score_sd = np.std(system_scores, ddof=1)
        human_score_sd = np.std(human_scores, ddof=1)

        if use_diff_std_means:

            # calculate the difference of standardized means
            smd_name = 'DSM'
            smd = difference_of_standardized_means(human_scores,
                                                   system_scores,
                                                   population_human_score_mn,
                                                   population_system_score_mn,
                                                   population_human_score_sd,
                                                   population_system_score_sd)

        else:

            # calculate the standardized mean difference
            smd_name = 'SMD'
            smd = standardized_mean_difference(human_scores,
                                               system_scores,
                                               population_human_score_sd,
                                               population_system_score_sd,
                                               method=smd_method)

        # compute r2
        if len(df) == 1:
            r2 = np.nan
        else:
            r2 = r2_score(human_scores, system_scores)

        # compute MSE
        mse = mean_squared_error(human_scores, system_scores)
        rmse = np.sqrt(mse)

        # return everything as a series
        metrics = pd.Series({'kappa': unweighted_kappa,
                             'wtkappa': weighted_kappa,
                             'exact_agr': human_system_agreement,
                             'adj_agr': human_system_adjacent_agreement,
                             smd_name: smd,
                             'corr': correlations,
                             'R2': r2,
                             'RMSE': rmse,
                             'sys_min': min_system_score,
                             'sys_max': max_system_score,
                             'sys_mean': mean_system_score,
                             'sys_sd': system_score_sd,
                             'h_min': min_human_score,
                             'h_max': max_human_score,
                             'h_mean': mean_human_score,
                             'h_sd': human_score_sd,
                             'N': len(system_scores)})

        return metrics

    @staticmethod
    def compute_disattenuated_correlations(human_machine_corr,
                                           human_human_corr):
        """
        Compute the disattenuated correlations between human and machine scores.
        These are computed as the Pearson's correlation between the human score
        and the machine score divided by the square root of correlation between
        two human raters.

        Parameters
        ----------
        human_machine_corr : pandas Series
            Series containing of pearson's correlation coefficients human-machine correlations
        human_human_corr : pandas Series
            Series containing of pearson's correlation coefficients for human-human correlations.
            This can contain a single value or have the index matching that of human-machine
            correlations

        Returns
        -------
        df_correlations: pandas DataFrame
            Data frame containing the HM correlation, HH correlation,
            and disattenuated correlation
        """

        # if we only have a single value for human correlation and the index
        # is not in human-machine values, we use the same HH value in all cases
        if len(human_human_corr) == 1 and not human_human_corr.index[0] in human_machine_corr.index:
            human_human_corr = pd.Series(human_human_corr.values.repeat(len(human_machine_corr)),
                                         index=human_machine_corr.index)

        # we now concatenate the two series on index
        df_correlations = pd.concat([human_machine_corr, human_human_corr],
                                    axis=1,
                                    sort=True,
                                    keys=['corr_HM', 'corr_HH'])

        # if any of the HH correlations are negative, we will ignore these
        # and treat them as Nones
        with np.errstate(invalid='ignore'):
            df_correlations['sqrt_HH'] = np.sqrt(df_correlations['corr_HH'])

        df_correlations['corr_disattenuated'] = (df_correlations['corr_HM'] /
                                                 df_correlations['sqrt_HH'])

        return df_correlations

    def compute_correlations_by_group(self,
                                      df,
                                      selected_features,
                                      target_variable,
                                      grouping_variable,
                                      include_length=False):
        """
        Compute various marginal and partial correlations of the given
        columns in the given data frame against the target variable for all
        data and for each level of the grouping variable.

        Parameters
        ----------
        df : pandas DataFrame
            Input data frame.
        selected_features : list of str
            List of feature names for which to compute
            the correlations.
        target_variable : str
            Feature name indicating the target variable i.e., the
            dependent variable
        grouping_variable : str
            Feature name that contain the grouping information
        include_length : bool, optional
            Whether or not to include the length when
            computing the partial correlations.
            Defaults to False.

        Returns
        -------
        df_output : pandas DataFrame
            Data frame containing the correlations.
        """

        df_desc = df.copy()

        columns = selected_features + [target_variable, grouping_variable]
        if include_length:
            columns.append('length')
        df_desc = df_desc[columns]

        # create a duplicate data frame to compute correlations
        # over the whole data, i.e., across all grouping variables
        df_desc_all = df_desc.copy()
        df_desc_all[grouping_variable] = 'All data'

        # combine the two data frames
        df_desc_combined = pd.concat([df_desc, df_desc_all], sort=True)
        df_desc_combined.reset_index(drop=True, inplace=True)

        # compute the various (marginal and partial) correlations with score
        ret = self.correlation_helper(df_desc_combined,
                                      target_variable,
                                      grouping_variable,
                                      include_length=include_length)

        return ret

    def filter_metrics(self,
                       df_metrics,
                       use_scaled_predictions=False,
                       chosen_metric_dict=None):
        """
        Filter the data frame `df_metrics` that contain all
        of the metric values by all score types (raw, raw_trim etc.)
        to retain only the metrics as defined in the given dictionary
        `chosen_metric_dict`. This is a dictionary that maps
        score types ('raw', 'scale', 'raw_trim' etc.)
        to the list of metrics that should be computed
        for them. The full list is ::

        - 'corr'
        - 'kappa'
        - 'wtkappa'
        - 'exact_agr'
        - 'adj_agr'
        - 'SMD' or `DSM`, depending on what is in `df_metrics`.
        - 'RMSE'
        - 'R2'
        - 'sys_min'
        - 'sys_max'
        - 'sys_mean'
        - 'sys_sd'
        - 'h_min'
        - 'h_max'
        - 'h_mean'
        - 'h_sd'
        - 'N'

        Parameters
        ----------
        df_metrics : pd.DataFrame
            The DataFrame to filter.
        use_scaled_predictions : bool, optional
            Whether to use scaled predictions.
            Defaults to False.
        chosen_metric_dict : dict, optional
            The dictionary to map score types to metrics that should be
            computer for them.
            Defaults to None.

        Notes
        -----
        Note that the last five metrics will be the same
        for all score types. If the dictionary is not specified
        then, the following dictionary, containing the recommended
        metrics, is used ::

           {'raw/scale_trim': ['N', 'h_mean', 'h_sd', 'sys_mean', 'sys_sd', 'wtkappa',
                               'corr', 'RMSE', 'R2', 'SMD'],
            'raw/scale_trim_round': ['sys_mean', 'sys_sd', 'kappa',
                                     'exact_agr', 'adj_agr', 'SMD']}

        where raw/scale is chosen depending on whether `use_scaled_predictions`
        is False or True.
        """
        # do we want the raw or the scaled metrics
        score_prefix = 'scale' if use_scaled_predictions else 'raw'

        # what metrics are we choosing to include?
        if chosen_metric_dict:
            chosen_metrics = chosen_metric_dict
        else:
            smd_name = 'DSM' if 'DSM' in df_metrics else 'SMD'
            chosen_metrics = {'{}_trim'.format(score_prefix): ['N',
                                                               'h_mean',
                                                               'h_sd',
                                                               'sys_mean',
                                                               'sys_sd',
                                                               'wtkappa',
                                                               'corr',
                                                               smd_name,
                                                               'RMSE',
                                                               'R2'],
                              '{}_trim_round'.format(score_prefix): ['sys_mean',
                                                                     'sys_sd',
                                                                     'kappa',
                                                                     'exact_agr',
                                                                     'adj_agr',
                                                                     smd_name]}

        # extract the metrics we need from the given metrics frame
        metricdict = {}
        for score_type in chosen_metrics:
            for metric in chosen_metrics[score_type]:
                colname = (metric if metric in ['h_mean', 'h_sd', 'N']
                           else '{}.{}'.format(metric, score_type))
                values = df_metrics[metric][score_type]
                metricdict[colname] = values

        df_filtered_metrics = pd.DataFrame([metricdict])
        return df_filtered_metrics

    def compute_metrics(self,
                        df,
                        compute_shortened=False,
                        use_scaled_predictions=False,
                        include_second_score=False,
                        population_sd_dict=None,
                        population_mn_dict=None,
                        smd_method='unpooled',
                        use_diff_std_means=False):
        """
        Compute the evaluation metrics for the scores in the given data frame.
        This function compute metrics for all score types.

        If `include_second_score` is True, then assume that a column called
        `sc2` containing a second human score is available and use that to
        compute the human-human evaluation stats and the performance
        degradation stats.

        If `compute_shortened` is set to True, then this function also
        computes a shortened version of the full human-machine metrics data
        frame. See `filter_metrics()` for a description of the default
        columns included in the shortened data frame.

        Parameters
        ----------
        df : pandas DataFrame
            Input data frame
        compute_shortened : bool, optional
            Also compute a shortened version of the full
            metrics data frame.
            Defaults to False.
        use_scaled_predictions : bool, optional
            Use evaluations based on scaled predictions in
            the shortened version of the metrics data frame.
            Defaults to False.
        include_second_score : bool, optional
            Second human score available.
            Defaults to False.
        population_sd_dict : dict, optional
            Dictionary containing population standard deviation for each column containing
            human or system scores. This is used to compute SMD for subgroups.
            Defaults to None.
        population_mn_dict : dict, optional
            Dictionary containing population mean for each column containing
            human or system scores. This is used to compute SMD for subgroups.
            Defaults to None.
        smd_method : {'williamson', 'johnson', pooled', 'unpooled'}, optional
            The SMD method to use, only used if `use_diff_std_means=False`.
            All methods have the same numerator mean(`y_pred`) - mean(`y_true_observed`)
            and the following denominators :

            - `williamson`: pooled population standard deviation of `y_true_observed` and `y_pred`
            - `johnson`: population standard deviation of `y_true_observed`.
            - `pooled`: pooled standard deviation of `y_true_observed` and `y_pred` for this group.
            - `unpooled`: standard deviation of `y_true_observed` for this group.

            Defaults to 'unpooled'.
        use_diff_std_means : bool, optional
            Whether to use the difference of standardized means, rather than the standardized mean
            difference. This is most useful with subgroup analysis.
            Defaults to False.

        Returns
        -------
        df_human_machine_eval : pandas DataFrame
            Data frame containing the full set of evaluation
            metrics.
        df_human_machine_eval_filtered : pandas DataFrame
            Data frame containing the human-human statistics
            but is empty if `include_second_score` is False.
        df_human_human_eval : pandas DataFrame
            A shortened version of the first data frame but
            is empty if `compute_shortened` is False.
        """

        # shorter variable name is easier to work with
        use_scaled = use_scaled_predictions

        # are we using DSM or SMD?
        smd_name = 'DSM' if use_diff_std_means else 'SMD'

        # get the population standard deviations for SMD if none were supplied
        if not population_sd_dict:
            population_sd_dict = {col: None for col in df.columns}

        # get the population standard deviations for SMD if none were supplied
        if not population_mn_dict:
            population_mn_dict = {col: None for col in df.columns}

        # if the second human score column is available, the values are
        # probably not available for all of the responses in the test
        # set and so we want to exclude 'sc2' from human-machine metrics
        # computation. In addition, we also want to compute the human-human
        # metrics only on the data that is double scored.
        df_human_human = pd.DataFrame()
        if include_second_score:

            df_single = df.drop('sc2', axis=1)

            df_human_machine = df_single.apply(lambda s:
                                               self.metrics_helper(df_single['sc1'],
                                                                   s,
                                                                   population_sd_dict['sc1'],
                                                                   population_sd_dict[s.name],
                                                                   population_mn_dict['sc1'],
                                                                   population_mn_dict[s.name],
                                                                   smd_method,
                                                                   use_diff_std_means))
            df_double = df[df['sc2'].notnull()][['sc1', 'sc2']]
            df_human_human = df_double.apply(lambda s:
                                             self.metrics_helper(df_double['sc1'],
                                                                 s,
                                                                 population_sd_dict['sc1'],
                                                                 population_sd_dict[s.name],
                                                                 population_mn_dict['sc1'],
                                                                 population_mn_dict[s.name],
                                                                 'pooled',
                                                                 use_diff_std_means))
            # drop the sc1 column from the human-human agreement frame
            df_human_human = df_human_human.drop('sc1', 1)

            # sort the rows in the correct order
            df_human_human = df_human_human.reindex(['N', 'h_mean', 'h_sd',
                                                          'h_min', 'h_max',
                                                          'sys_mean', 'sys_sd',
                                                          'sys_min', 'sys_max',
                                                          'corr', 'wtkappa', 'R2',
                                                          'kappa', 'exact_agr',
                                                          'adj_agr', smd_name, 'RMSE'])
            # rename `h_*` -> `h1_*` and `sys_*` -> `h2_*`
            df_human_human.rename(lambda c: c.replace('h_', 'h1_').replace('sys_', 'h2_'),
                                  inplace=True)
            # drop RMSE and R2 because they are not meaningful for human raters
            df_human_human.drop(['R2', 'RMSE'], inplace=True)
            df_human_human = df_human_human.transpose()
            # convert N to integer if it's not empty else set to 0
            try:
                df_human_human['N'] = df_human_human['N'].astype(int)
            except ValueError:
                df_human_human['N'] = 0
            df_human_human.index = ['']
        else:
            df_human_machine = df.apply(lambda s: self.metrics_helper(df['sc1'],
                                                                      s,
                                                                      population_sd_dict['sc1'],
                                                                      population_sd_dict[s.name],
                                                                      population_mn_dict['sc1'],
                                                                      population_mn_dict[s.name],
                                                                      smd_method,
                                                                      use_diff_std_means))

        # drop 'sc1' column from the human-machine frame and transpose
        df_human_machine = df_human_machine.drop('sc1', 1)
        df_human_machine = df_human_machine.transpose()

        # sort the columns and rows in the correct order
        df_human_machine = df_human_machine[['N',
                                             'h_mean', 'h_sd',
                                             'h_min', 'h_max',
                                             'sys_mean', 'sys_sd',
                                             'sys_min', 'sys_max',
                                             'corr',
                                             'wtkappa', 'R2', 'kappa',
                                             'exact_agr', 'adj_agr',
                                             smd_name, 'RMSE']]

        # make N column an integer if it's not NaN else set it to 0
        df_human_machine['N'] = df_human_machine['N'].astype(int)
        all_rows_order = ['raw', 'raw_trim', 'raw_trim_round',
                          'scale', 'scale_trim', 'scale_trim_round']
        existing_rows_index = [row for row in all_rows_order if row in df_human_machine.index]
        df_human_machine = df_human_machine.reindex(existing_rows_index)

        # extract some default metrics for a shorter version of this data frame
        # if we were asked to do so
        if compute_shortened:
            df_human_machine_filtered = self.filter_metrics(df_human_machine,
                                                            use_scaled_predictions=use_scaled)
        else:
            df_human_machine_filtered = pd.DataFrame()

        # return all data frames
        return (df_human_machine,
                df_human_machine_filtered,
                df_human_human)

    def compute_metrics_by_group(self,
                                 df_test,
                                 grouping_variable,
                                 use_scaled_predictions=False,
                                 include_second_score=False):
        """
        Compute a subset of the evaluation metrics for the scores
        in the given data frame by group specified in `grouping_variable`.
        See `filter_metrics()` above for a description of the subset
        that is selected.

        Parameters
        ----------
        df_test : pandas DataFrame
            Input data frame.
        grouping_variable : str
            Feature name indicating the column that
            contains grouping information.
        use_scaled_predictions : bool, optional
            Include scaled predictions when computing
            the evaluation metrics, defaults to False.
        include_second_score : bool, optional
            Include human-human association statistics,
            defaults to False.

        Returns
        -------
        df_human_machine_eval_by_group : pandas DataFrame
            Data frame containing the correlation
            human-machine association statistics.
        df_human_human_eval_by_group : pandas DataFrame
            Data frame that either contains the human-human
            statistics or is an empty data frame, depending
            on whether `include_second_score` is True.
        """

        # get the population standard deviation that we will need to compute SMD for all columns
        # other than id and subgroup
        population_sd_dict = {col: df_test[col].std(ddof=1)
                              for col in df_test.columns if col not in ['spkitemid',
                                                                        grouping_variable]}

        population_mn_dict = {col: df_test[col].mean()
                              for col in df_test.columns if col not in ['spkitemid',
                                                                        grouping_variable]}

        # check if any of the standard deviations is zero and
        # tell user to expect to see many warnings.
        zero_sd_scores = [score for (score, sd) in population_sd_dict.items() if
                          np.isclose(sd, 0, atol=1e-07)]
        if len(zero_sd_scores) > 0:
            warnings.warn("The standard deviation for {} scores "
                          "is zero (all values are the same). You "
                          "will see multiple warnings about DSM computation "
                          "since this metric is computed separately for "
                          "each subgroup.".format(', '.join(zero_sd_scores)))

        # create a duplicate data frame to compute evaluations
        # over the whole data, i.e., across groups
        df_preds_all = df_test.copy()
        df_preds_all[grouping_variable] = 'All data'

        # combine the two data frames
        df_preds_combined = pd.concat([df_test, df_preds_all], sort=True)
        df_preds_combined.reset_index(drop=True, inplace=True)

        # group by the grouping_variable columns
        grouped = df_preds_combined.groupby(grouping_variable)

        df_human_machine_by_group = pd.DataFrame()
        df_human_human_by_group = pd.DataFrame()

        for group, df_group in grouped:
            df_group = df_group.drop(grouping_variable, 1)

            (df_human_machine_metrics,
             df_human_machine_metrics_short,
             df_human_human_metrics
             ) = self.compute_metrics(df_group,
                                      compute_shortened=True,
                                      use_scaled_predictions=use_scaled_predictions,
                                      include_second_score=include_second_score,
                                      population_sd_dict=population_sd_dict,
                                      population_mn_dict=population_mn_dict,
                                      use_diff_std_means=True)

            # we need to convert the shortened data frame to a series here
            df_human_machine_by_group[group] = df_human_machine_metrics_short.iloc[0]

            # update the by group human-human metrics frame if
            # we have the second score column available
            if include_second_score:
                df_human_human_metrics.index = [group]
                df_human_human_by_group = df_human_human_by_group.append(df_human_human_metrics)

        # transpose the by group human-machine metrics frame
        df_human_machine_by_group = df_human_machine_by_group.transpose()

        return (df_human_machine_by_group, df_human_human_by_group)

    def compute_degradation_and_disattenuated_correlations(self,
                                                           df,
                                                           use_all_responses=True):
        """
        Compute the degradation in performance when using the machine
        to predict the score instead of a second human and the
        the disattenuated correlations between human and machine scores.
        These are computed as the Pearson's correlation between the human score
        and the machine score divided by the square root of correlation between
        two human raters.

        For this, we can compute the machine performance either only on the
        double scored data or on the full dataset. Both options have their
        pros and cons. The default is to use the full dataset. This function
        also assumes that the `sc2` column exists in the given data frame,
        in addition to `sc1` and the various types of predictions.

        Parameters
        ----------
        df : pandas DataFrame
            Input data frame.
        use_all_responses : bool, optional
            Use the full data set instead of only using
            the double-scored subset, defaults to True.

        Returns
        -------
        df_degradation : pandas DataFrame
            Data frame containing the degradation statistics.

        df_correlations : pandas DataFrame
            Data frame containing the HM correlation, HH correlation
            and disattenuated correlation
        """

        if use_all_responses:
            df_responses = df
        else:
            # use only double scored data
            df_responses = df[df['sc2'].notnull()]

        # compute the human-machine and human-human metrics
        (df_human_machine_eval,
         _,
         df_human_human_eval) = self.compute_metrics(df_responses,
                                                     include_second_score=True)

        # compute disattenuated correlations
        df_correlations = self.compute_disattenuated_correlations(df_human_machine_eval['corr'],
                                                                  df_human_human_eval['corr'])

        # Compute degradation. we only care about the degradation in these metrics
        degradation_metrics = ['corr', 'kappa', 'wtkappa',
                               'exact_agr', 'adj_agr', 'SMD']
        df_human_machine_eval = df_human_machine_eval[degradation_metrics]
        df_human_human_eval = df_human_human_eval[degradation_metrics]
        df_degradation = df_human_machine_eval.apply(lambda row:
                                                     row - df_human_human_eval.loc[''], axis=1)

        return (df_degradation, df_correlations)

    def run_training_analyses(self,
                              data_container,
                              configuration):
        """
        Run all of the analyses on the training data.

        Parameters
        ----------
        data_container : container.DataContainer
            The DataContainer object. This container must include the following
            DataFrames: {'train_features', 'train_metadata','train_preprocessed_features',
            'train_length', 'train_features'}

        configuration : configuration_parser.Configuration
            The Configuration object.  This configuration object mus include the
            following parameters (keys) ::

                {'length_column', 'subgroups', 'selected_features'}

        Returns
        -------
        data_container : container.DataContainer
            A new DataContainer object with the following DataFrames ::
            - feature_descriptives
            - feature_descriptivesExtra
            - feature_outliers
            - cors_orig
            - cors_processed
            - margcor_score_all_data
            - pcor_score_all_data
            - pcor_score_no_length_all_data
            - margcor_length_all_data
            - pcor_length_all_data
            - pca
            - pcavar
            - margcor_length_by_*
            - pcor_length_by_*
            - margcor_score_by_*
            - pcor_score_by_*
            - pcor_score_no_length_by_*

        configuration : configuration_parser.Configuration
            A new Configuration object
        """

        frame_names = ['train_features', 'train_metadata',
                       'train_preprocessed_features', 'train_length',
                       'train_features']

        param_names = ['length_column', 'subgroups',
                       'selected_features']

        self.check_frame_names(data_container,
                               frame_names)

        self.check_param_names(configuration,
                               param_names)

        # only use the features selected by the model but keep their order the same
        # as in the original file as ordering may affect the sign in pca
        df_train = data_container.train_features.copy()
        df_train_length = data_container.train_length.copy()
        df_train_metadata = data_container.train_metadata.copy()
        df_train_preprocessed_features = data_container.train_preprocessed_features.copy()

        subgroups = configuration['subgroups']
        selected_features = configuration['selected_features']

        df_train_preprocessed = pd.merge(df_train_preprocessed_features,
                                         df_train_metadata, on='spkitemid')

        assert (len(df_train_preprocessed.index) ==
                len(df_train_preprocessed_features.index) ==
                len(df_train_metadata.index))

        # get descriptives, percentiles and outliers for the original feature values
        df_descriptives = self.compute_basic_descriptives(df_train, selected_features)
        df_percentiles = self.compute_percentiles(df_train, selected_features)
        df_outliers = self.compute_outliers(df_train, selected_features)

        # set a general boolean flag indicating if we should include length
        include_length = not df_train_length.empty

        # include length if available
        if include_length:
            columns = selected_features + ['sc1', 'length']
            df_train_with_length = df_train.merge(df_train_length, on='spkitemid')
            df_train_preprocess_length = df_train_preprocessed.merge(df_train_length,
                                                                     on='spkitemid')
        else:
            columns = selected_features + ['sc1']
            df_train_with_length = df_train
            df_train_preprocess_length = df_train_preprocessed

        # get pairwise correlations against the original training features
        # as well as the pre-processed training features
        df_pairwise_cors_orig = df_train_with_length[columns].corr(method='pearson')
        df_pairwise_cors_preprocess = df_train_preprocess_length[columns].corr(method='pearson')

        # get marginal and partial correlations against sc1 for all data
        # for partial correlations, we partial out all other features
        df_train_with_group_for_all = df_train_preprocess_length.copy()
        df_train_with_group_for_all = df_train_with_group_for_all[columns]
        df_train_with_group_for_all['all_data'] = 'All data'

        (df_margcor_sc1,
         df_pcor_sc1,
         df_pcor_sc1_no_length) = self.correlation_helper(df_train_with_group_for_all,
                                                          'sc1',
                                                          'all_data',
                                                          include_length=include_length)

        # get marginal and partial correlations against length for all data
        # if the length column is available
        df_margcor_length = pd.DataFrame()
        df_pcor_length = pd.DataFrame()

        if include_length:

            df_train_with_group_for_all = df_train_preprocess_length.copy()
            columns = selected_features + ['length']

            df_train_with_group_for_all = df_train_with_group_for_all[columns]
            df_train_with_group_for_all['all_data'] = 'All data'

            (df_margcor_length,
             df_pcor_length,
             _) = self.correlation_helper(df_train_with_group_for_all,
                                          'length',
                                          'all_data')

        # get marginal and partial correlations against sc1 by group (preprocessed features)
        # also include partial correlations with length if length is available
        score_corr_by_group_dict = {}
        include_length = 'length' in df_train_preprocess_length
        for grouping_variable in subgroups:

            corr_by_group = self.compute_correlations_by_group(df_train_preprocess_length,
                                                               selected_features,
                                                               'sc1',
                                                               grouping_variable,
                                                               include_length=include_length)

            score_corr_by_group_dict[grouping_variable] = corr_by_group

        # get marginal and partial correlations against sc1 by group (preprocessed features)
        length_corr_by_group_dict = {}
        if include_length:
            for grouping_variable in subgroups:

                corr_by_group = self.compute_correlations_by_group(df_train_preprocess_length,
                                                                   selected_features,
                                                                   'length',
                                                                   grouping_variable)

                length_corr_by_group_dict[grouping_variable] = corr_by_group

        # get PCA information
        df_pca_components, df_pca_variance = self.compute_pca(df_train_preprocessed,
                                                              selected_features)

        # Datasets to add
        datasets = [{'name': 'feature_descriptives', 'frame': df_descriptives},
                    {'name': 'feature_descriptivesExtra', 'frame': df_percentiles},
                    {'name': 'feature_outliers', 'frame': df_outliers},
                    {'name': 'cors_orig', 'frame': df_pairwise_cors_orig},
                    {'name': 'cors_processed', 'frame': df_pairwise_cors_preprocess},
                    {'name': 'margcor_score_all_data', 'frame': df_margcor_sc1},
                    {'name': 'pcor_score_all_data', 'frame': df_pcor_sc1},
                    {'name': 'pcor_score_no_length_all_data', 'frame': df_pcor_sc1_no_length},
                    {'name': 'margcor_length_all_data', 'frame': df_margcor_length},
                    {'name': 'pcor_length_all_data', 'frame': df_pcor_length},
                    {'name': 'pca', 'frame': df_pca_components},
                    {'name': 'pcavar', 'frame': df_pca_variance}]

        # Add length correlation by group datasets
        for group in length_corr_by_group_dict:

            (length_marg_cors,
             length_part_cors,
             _) = length_corr_by_group_dict.get(group,
                                                (pd.DataFrame(),
                                                 pd.DataFrame(),
                                                 pd.DataFrame()))

            datasets.extend([{'name': 'margcor_length_by_{}'.format(group),
                              'frame': length_marg_cors},
                             {'name': 'pcor_length_by_{}'.format(group),
                              'frame': length_part_cors}])

        # Add score correlations by group datasets
        for group in score_corr_by_group_dict:

            (sc1_marg_cors,
             sc1_part_cors,
             sc1_part_cors_no_length) = score_corr_by_group_dict[group]

            datasets.extend([{'name': 'margcor_score_by_{}'.format(group),
                              'frame': sc1_marg_cors},
                             {'name': 'pcor_score_by_{}'.format(group),
                              'frame': sc1_part_cors},
                             {'name': 'pcor_score_no_length_by_{}'.format(group),
                              'frame': sc1_part_cors_no_length}])

        return configuration, DataContainer(datasets=datasets)

    def run_prediction_analyses(self,
                                data_container,
                                configuration):
        """
        Run all the analyses on the machine predictions.

        Parameters
        ----------
        data_container : container.DataContainer
            The DataContainer object. This container must include the following
            DataFrames: {'train_features', 'train_metadata','train_preprocessed_features',
            'train_length', 'train_features'}

        configuration : configuration_parser.Configuration
            The Configuration object.  This configuration object must include the
            following parameters (keys) ::

                {'subgroups', 'second_human_score_column',
                 'use_scaled_predictions'}

        Returns
        -------
        data_container : container.DataContainer
            A new DataContainer object with the following DataFrames ::

            - eval
            - eval_short
            - consistency
            - degradation
            - disattenudated_correlations
            - confMatrix
            - score_dist
            - eval_by_*
            - consistency_by_*
            - disattenduated_correlations_by_*
            - true_score_eval

        configuration : configuration_parser.Configuration
            A new Configuration object.
        """

        frame_names = ['pred_test', 'test_metadata',
                       'test_human_scores']

        param_names = ['subgroups', 'second_human_score_column',
                       'use_scaled_predictions']

        self.check_frame_names(data_container,
                               frame_names)

        self.check_param_names(configuration,
                               param_names)

        df_test = data_container.pred_test.copy()
        df_test_metadata = data_container.test_metadata.copy()
        df_test_human_scores = data_container.test_human_scores.copy()

        subgroups = configuration['subgroups']
        use_scaled_predictions = configuration['use_scaled_predictions']

        df_preds = pd.merge(df_test, df_test_metadata, on='spkitemid')

        assert len(df_preds.index) == len(df_test.index) == len(df_test_metadata.index)

        # set a general boolean flag indicating if
        # we should include the second human score
        include_second_score = not df_test_human_scores.empty

        # extract the columns that contain predictions
        prediction_columns = [column for column in df_test if column != 'spkitemid']

        # if a second score is available, use it
        if include_second_score:
            prediction_columns.append('sc2')
            df_preds_second_score = df_preds.merge(df_test_human_scores[['spkitemid', 'sc2']],
                                                   on='spkitemid')
        else:
            df_preds_second_score = df_preds

        # compute the evaluation metrics over the whole data set
        (df_human_machine,
         df_human_machine_short,
         df_human_human) = self.compute_metrics(df_preds_second_score[prediction_columns],
                                                compute_shortened=True,
                                                use_scaled_predictions=use_scaled_predictions,
                                                include_second_score=include_second_score)

        # compute the evaluation metrics by group
        eval_by_group_dict = {}
        for group in subgroups:
            group_columns = prediction_columns + [group]
            metrics = self.compute_metrics_by_group(df_preds_second_score[group_columns],
                                                    group,
                                                    use_scaled_predictions=use_scaled_predictions,
                                                    include_second_score=include_second_score)
            eval_by_group_dict[group] = metrics

        # compute the degradation statistics and disattenuated correlations
        # if we have the second human score available
        df_degradation = pd.DataFrame()
        df_correlations = pd.DataFrame()
        if include_second_score:
            (df_degradation,
             df_correlations) = self.compute_degradation_and_disattenuated_correlations(df_preds_second_score[prediction_columns])

        # the following two evaluations require rounded human scores
        # we create a column for this
        df_preds['sc1_round'] = np.round(df_preds['sc1'])

        # compute the confusion matrix as a data frame
        score_type = 'scale' if use_scaled_predictions else 'raw'
        human_scores = df_preds['sc1_round'].astype('int64')
        system_scores = df_preds['{}_trim_round'.format(score_type)].astype('int64')
        conf_matrix = confusion_matrix(human_scores, system_scores)
        labels = sorted(human_scores.append(system_scores).unique())
        df_confmatrix = pd.DataFrame(conf_matrix, index=labels, columns=labels).transpose()

        # compute the score distributions of the rounded human and machine scores
        df_score_dist = df_preds[['sc1_round',
                                  '{}_trim_round'
                                  ''.format(score_type)]].apply(lambda s:
                                                                (s.value_counts() /
                                                                 len(df_test) * 100))

        # Replace any NaNs, which we might get because our model never
        # predicts a particular score label, with zeros.
        df_score_dist.fillna(0, inplace=True)

        df_score_dist.columns = ['human', 'sys_{}'.format(score_type)]
        df_score_dist['difference'] = (df_score_dist['sys_{}'.format(score_type)] -
                                       df_score_dist['human'])
        df_score_dist['score'] = df_score_dist.index

        df_score_dist = df_score_dist[['score', 'human',
                                       'sys_{}'.format(score_type),
                                       'difference']]
        df_score_dist.sort_values(by='score', inplace=True)

        datasets = [{'name': 'eval', 'frame': df_human_machine},
                    {'name': 'eval_short', 'frame': df_human_machine_short},
                    {'name': 'consistency', 'frame': df_human_human},
                    {'name': 'degradation', 'frame': df_degradation},
                    {'name': 'disattenuated_correlations', 'frame': df_correlations},
                    {'name': 'confMatrix', 'frame': df_confmatrix},
                    {'name': 'score_dist', 'frame': df_score_dist}]

        # compute true-score analyses if we have second score
        # or have been given rater error variance
        rater_error_variance = configuration.get_rater_error_variance()

        if include_second_score or rater_error_variance is not None:
            system_score_columns = [col for col in prediction_columns
                                    if col not in ['sc1', 'sc2']]

            human_score_columns = [col for col in prediction_columns
                                   if col in ['sc1', 'sc2']]

            df_prmse = get_true_score_evaluations(df_preds_second_score,
                                                  system_score_columns,
                                                  human_score_columns,
                                                  rater_error_variance)

            datasets.extend([{'name': 'true_score_eval', 'frame': df_prmse}])

        for group in eval_by_group_dict:
            eval_by_group, consistency_by_group = eval_by_group_dict[group]

            # compute disattenuated correlations if we have the second human score
            if include_second_score:
                dis_corr_by_group = self.compute_disattenuated_correlations(eval_by_group['corr.{}_trim'.format(score_type)],
                                                                            consistency_by_group['corr'])
            else:
                dis_corr_by_group = pd.DataFrame()

            datasets.extend([{'name': 'eval_by_{}'.format(group),
                              'frame': eval_by_group},
                             {'name': 'consistency_by_{}'.format(group),
                              'frame': consistency_by_group},
                             {'name': 'disattenuated_correlations_by_{}'.format(group),
                              'frame': dis_corr_by_group}])

        return configuration, DataContainer(datasets=datasets)

    def run_data_composition_analyses_for_rsmtool(self,
                                                  data_container,
                                                  configuration):
        """
        Run all data composition analyses for RSMTool.

        Parameters
        ----------
        data_container : container.DataContainer
            The DataContainer object. This container must include the following
            DataFrames: {'test_metadata', 'train_metadata','train_excluded',
            'test_excluded', 'train_features'}
        configuration : configuration_parser.Configuration
            The Configuration object.  This configuration object must include the
            following parameters (keys) ::

                {'subgroups', 'candidate_column',
                 'exclude_zero_scores', 'exclude_listwise'}

        Returns
        -------
        data_container : container.DataContainer
            A new DataContainer object with the following DataFrames ::
            - test_excluded_composition
            - train_excluded_composition
            - data_composition
            - data_composition_by_*
        configuration : configuration_parser.Configuration
            A new Configuration object.
        """

        frame_names = ['train_metadata', 'test_metadata',
                       'train_excluded', 'test_excluded',
                       'train_features']

        param_names = ['candidate_column', 'subgroups',
                       'exclude_zero_scores', 'exclude_listwise']

        self.check_frame_names(data_container,
                               frame_names)

        self.check_param_names(configuration,
                               param_names)

        features = [column for column in data_container.train_features.columns
                    if column not in ['spkitemid', 'sc1']]

        exclude_scores = configuration['exclude_zero_scores']
        exclude_listwise = configuration['exclude_listwise']

        subgroups = configuration['subgroups']
        candidate_column = configuration['candidate_column']

        df_train_excluded = self.analyze_excluded_responses(data_container['train_excluded'],
                                                            features,
                                                            'Score/Features',
                                                            exclude_zero_scores=exclude_scores,
                                                            exclude_listwise=exclude_listwise)

        df_test_excluded = self.analyze_excluded_responses(data_container['test_excluded'],
                                                           features,
                                                           'Score/Features',
                                                           exclude_zero_scores=exclude_scores,
                                                           exclude_listwise=exclude_listwise)

        df_data_composition = self.analyze_used_responses(data_container['train_metadata'],
                                                          data_container['test_metadata'],
                                                          subgroups, candidate_column)

        # do the analysis by subgroups
        # first create a joint data frame with both sets
        df_train_metadata_with_set = data_container['train_metadata'].copy()
        df_test_metadata_with_set = data_container['test_metadata'].copy()

        df_train_metadata_with_set['set'] = 'Training set'
        df_test_metadata_with_set['set'] = 'Evaluation set'

        df_both_metadata = pd.merge(df_train_metadata_with_set,
                                    df_test_metadata_with_set,
                                    how='outer')

        # create contingency table for each subgroup
        data_composition_by_group_dict = {}
        for grouping_variable in subgroups:
            df_crosstab_group = pd.crosstab(df_both_metadata[grouping_variable],
                                            df_both_metadata['set'])
            df_crosstab_group = df_crosstab_group[['Training set',
                                                   'Evaluation set']]
            df_crosstab_group.insert(0, grouping_variable, df_crosstab_group.index)
            data_composition_by_group_dict[grouping_variable] = df_crosstab_group

        datasets = [{'name': 'test_excluded_composition', 'frame': df_test_excluded},
                    {'name': 'train_excluded_composition', 'frame': df_train_excluded},
                    {'name': 'data_composition', 'frame': df_data_composition}]

        for group in data_composition_by_group_dict:

            datasets.append({'name': 'data_composition_by_{}'.format(group),
                             'frame': data_composition_by_group_dict[group]})

        return configuration, DataContainer(datasets=datasets)

    def run_data_composition_analyses_for_rsmeval(self,
                                                  data_container,
                                                  configuration):
        """
        Similar to `run_data_composition_analyses_for_rsmtool()`
        but for RSMEval.

        Parameters
        ----------
        data_container : container.DataContainer
            The DataContainer object. This container must include the following
            DataFrames: {'test_metadata', 'test_excluded'}

        configuration : configuration_parser.Configuration
            The Configuration object.  This configuration object must include the
            following parameters (keys) ::

                {'subgroups', 'candidate_column',
                 'exclude_zero_scores', 'exclude_listwise'}

        Returns
        -------
        data_container : container.DataContainer
            A new DataContainer object with the following DataFrames ::
            - test_excluded_composition
            - data_composition
            - data_composition_by_*

        configuration : configuration_parser.Configuration
            A new Configuration object.
        """

        frame_names = ['test_metadata',
                       'test_excluded']

        param_names = ['candidate_column', 'subgroups',
                       'exclude_zero_scores', 'exclude_listwise']

        self.check_frame_names(data_container,
                               frame_names)

        self.check_param_names(configuration,
                               param_names)

        exclude_scores = configuration['exclude_zero_scores']
        exclude_listwise = configuration['exclude_listwise']

        subgroups = configuration['subgroups']
        candidate_column = configuration['candidate_column']

        # analyze excluded responses
        df_test_excluded = self.analyze_excluded_responses(data_container['test_excluded'],
                                                           ['raw'],
                                                           'Human/System',
                                                           exclude_zero_scores=exclude_scores,
                                                           exclude_listwise=exclude_listwise)

        # rename the columns and index in the analysis data frame
        df_test_excluded.rename(columns={'all features numeric':
                                         'numeric system score',
                                         'non-numeric feature values':
                                         'non-numeric system score'},

                                inplace=True)
        df_data_composition = self.analyze_used_predictions(data_container['test_metadata'],
                                                            subgroups,
                                                            candidate_column)

        # create contingency table for each group
        data_composition_by_group_dict = {}
        for grouping_variable in subgroups:
            series_crosstab_group = pd.pivot_table(data_container['test_metadata'],
                                                   values='spkitemid',
                                                   index=[grouping_variable],
                                                   aggfunc=len)

            df_crosstab_group = pd.DataFrame(series_crosstab_group)
            df_crosstab_group.insert(0, grouping_variable, df_crosstab_group.index)
            df_crosstab_group.rename(columns={'spkitemid': 'N responses'},
                                     inplace=True)
            data_composition_by_group_dict[grouping_variable] = df_crosstab_group

        datasets = [{'name': 'test_excluded_composition', 'frame': df_test_excluded},
                    {'name': 'data_composition', 'frame': df_data_composition}]

        for group in data_composition_by_group_dict:

            datasets.append({'name': 'data_composition_by_{}'.format(group),
                             'frame': data_composition_by_group_dict[group]})

        return configuration, DataContainer(datasets=datasets)
