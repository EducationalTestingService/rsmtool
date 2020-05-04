"""
Class for dealing with training built-in or SKLL models,
as well as making predictions for new data.

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

import logging
import pickle

from math import log10, sqrt
from os.path import join

import numpy as np
import pandas as pd
import statsmodels.api as sm

from numpy.random import RandomState
from scipy.optimize import nnls
from sklearn.linear_model import LassoCV
from skll import FeatureSet, Learner

from .analyzer import Analyzer
from .container import DataContainer
from .utils.metrics import compute_expected_scores_from_model
from .utils.models import is_skll_model
from .preprocessor import FeaturePreprocessor
from .writer import DataWriter


class Modeler:
    """
    A class for training and predicting with either
    built-in or SKLL models. Also provides helper functions
    for predicting train and test datasets.
    """

    def __init__(self):

        self.learner = None

    @classmethod
    def load_from_file(cls, model_path):
        """
        Load a Model object from file.

        Parameters
        ----------
        model_path : str
            The path to a model

        Returns
        -------
        model : Modeler
            A Modeler instance

        Raises
        ------
        ValuError
            If the `model_path` does not end with '.model'
        """
        if not model_path.lower().endswith('.model'):
            raise ValueError('The file `{}` does not end with the '
                             'proper extension. Please make sure that '
                             'it is a `.model` file.'.format(model_path))

        # Create SKLL learner from file
        learner = Learner.from_file(model_path)
        return cls.load_from_learner(learner)

    @classmethod
    def load_from_learner(cls, learner):
        """
        Load a Modeler object from file.

        Parameters
        ----------
        learner : SKLL.Learner
            A SKLL Learner object

        Returns
        -------
        modeler : Modeler
            A Modeler instance

        Raises
        ------
        TypeError
            If `learner` is not SKLL.Learner instance.
        """
        if not isinstance(learner, Learner):
            raise TypeError('The `learner` argument must be a '
                            '` SKLL.Learner` instance, not `{}`.'
                            ''.format(type(learner)))

        # Create Modeler instance
        modeler = Modeler()
        modeler.learner = learner
        return modeler

    @staticmethod
    def model_fit_to_dataframe(fit):
        """
        Take an object containing a statsmodels OLS model fit and extact
        the main model fit metrics into a data frame.

        Parameters
        ----------
        fit : a statsmodels fit object
            Model fit object obtained from a linear model trained using
            `statsmodels.OLS`.

        Returns
        -------
        df_fit : pandas DataFrame
            Data frame with the main model fit metrics.
        """

        df_fit = pd.DataFrame({"N responses": [int(fit.nobs)]})
        df_fit['N features'] = int(fit.df_model)
        df_fit['R2'] = fit.rsquared
        df_fit['R2_adjusted'] = fit.rsquared_adj
        return df_fit

    @staticmethod
    def ols_coefficients_to_dataframe(coefs):
        """
        Take a series containing OLS coefficients and convert it
        to a data frame.

        Parameters
        ----------
        coefs : pandas Series
            Series with feature names in the index and the coefficient
            values as the data, obtained from a linear model trained
            using `statsmodels.OLS`.

        Returns
        -------
        df_coef : pandas DataFrame
            Data frame with two columns, the first being the feature name
            and the second being the coefficient value.

        Note
        ----
        The first row in the output data frame is always for the intercept
        and the rest are sorted by feature name.
        """
        # first create a sorted data frame for all the non-intercept features
        non_intercept_columns = [c for c in coefs.index if c != 'const']
        df_non_intercept = pd.DataFrame(coefs.filter(non_intercept_columns),
                                        columns=['coefficient'])
        df_non_intercept.index.name = 'feature'
        df_non_intercept = df_non_intercept.sort_index()
        df_non_intercept.reset_index(inplace=True)

        # now create a data frame that just has the intercept
        df_intercept = pd.DataFrame([{'feature': 'Intercept',
                                      'coefficient': coefs['const']}])

        # append the non-intercept frame to the intercept one
        df_coef = df_intercept.append(df_non_intercept, sort=True, ignore_index=True)

        # we always want to have the feature column first
        df_coef = df_coef[['feature', 'coefficient']]

        return df_coef

    @staticmethod
    def skll_learner_params_to_dataframe(learner):
        """
        Take the given SKLL learner object and return a data
        frame containing its parameters.

        Parameters
        ----------
        learner : SKLL.Learner
            A SKLL learner object

        Returns
        -------
        df_coef : pandas DataFrame
            a data frame containing the model parameters
            from the given SKLL learner object.

        Note
        ----
        1. We use underlying `sklearn` model object to get at the
        coefficients and the intercept because the `model_params` attribute
        of the SKLL model ignores zero coefficients, which we do not want.
        2. The first row in the output data frame is always for the intercept
        and the rest are sorted by feature name.
        """
        # get the intercept, coefficients, and feature names
        intercept = learner.model.intercept_
        coefficients = learner.model.coef_
        feature_names = learner.feat_vectorizer.get_feature_names()

        # first create a sorted data frame for all the non-intercept features
        df_non_intercept = pd.DataFrame({'feature': feature_names,
                                         'coefficient': coefficients})
        df_non_intercept = df_non_intercept.sort_values(by=['feature'])

        # now create a data frame that just has the intercept
        df_intercept = pd.DataFrame([{'feature': 'Intercept',
                                      'coefficient': intercept}])

        # append the non-intercept frame to the intercept one
        df_coef = df_intercept.append(df_non_intercept, sort=True, ignore_index=True)

        # we always want to have the feature column first
        df_coef = df_coef[['feature', 'coefficient']]

        return df_coef

    @staticmethod
    def create_fake_skll_learner(df_coefficients):
        """
        Create fake SKLL linear regression learner object
        using the coefficients in the given data frame.

        Parameters
        ----------
        df_coefficients : pandas DataFrame
            Data frame containing the linear coefficients
            we want to create the fake SKLL model with.

        Returns
        -------
        learner: skll Learner object
            SKLL LinearRegression Learner object containing
            with the specified coefficients.
        """

        # initialize a random number generator
        randgen = RandomState(1234567890)

        # iterate over the coefficients
        coefdict = {}
        for feature, coefficient in df_coefficients.itertuples(index=False):
            if feature == 'Intercept':
                intercept = coefficient
            else:
                # exclude NA coefficients
                if coefficient == np.nan:
                    logging.warning("No coefficient was estimated for "
                                    "{}. This is likely due to exact "
                                    "collinearity in the model. This "
                                    "feature will not be used for model "
                                    "building".format(feature))
                else:
                    coefdict[feature] = coefficient

        learner = Learner('LinearRegression')
        num_features = len(coefdict)  # excluding the intercept
        fake_feature_values = randgen.rand(num_features)
        fake_features = [dict(zip(coefdict, fake_feature_values))]
        fake_fs = FeatureSet('fake', ids=['1'], labels=[1.0], features=fake_features)
        learner.train(fake_fs, grid_search=False)

        # now create its parameters from the coefficients from the built-in model
        learner.model.coef_ = learner.feat_vectorizer.transform(coefdict).toarray()[0]
        learner.model.intercept_ = intercept
        return learner

    def train_linear_regression(self, df_train, feature_columns):
        """
        Train `LinearRegression` (formerly empWt) -
        A simple linear regression model.

        Parameters
        ----------
        df_train : pd.DataFrame
            Data frame containing the features on which
            to train the model.
        feature_columns : list
            A list of feature columns to use in training the model.

        Returns
        -------
        learner : skll.Learner
            The SKLL learner object
        fit : statsmodels.RegressionResults
            A statsmodels regression results object.
        df_coef : pd.DataFrame
            The model coefficients in a data_frame
        used_features : list
            A list of features used in the final model.
        """
        # get the feature columns
        X = df_train[feature_columns]

        # add the intercept
        X = sm.add_constant(X)

        # fit the model
        fit = sm.OLS(df_train['sc1'], X).fit()
        df_coef = self.ols_coefficients_to_dataframe(fit.params)
        learner = self.create_fake_skll_learner(df_coef)

        # we used all the features
        used_features = feature_columns

        return learner, fit, df_coef, used_features

    def train_equal_weights_lr(self, df_train, feature_columns):
        """
        Train `EqualWeightsLR` (formerly eqWt) -
        All features get equal weight.

        Parameters
        ----------
        df_train : pd.DataFrame
            Data frame containing the features on which
            to train the model.
        feature_columns : list
            A list of feature columns to use in training the model.

        Returns
        -------
        learner : skll.Learner
            The SKLL learner object
        fit : statsmodels.RegressionResults
            A statsmodels regression results object.
        df_coef : pd.DataFrame
            The model coefficients in a data_frame
        used_features : list
            A list of features used in the final model.
        """
        # we first compute a single feature that is simply the sum of all features
        df_train_eqwt = df_train.copy()
        df_train_eqwt['sumfeature'] = df_train_eqwt[feature_columns].apply(np.sum,
                                                                           axis=1)

        # train a plain Linear Regression model
        X = df_train_eqwt['sumfeature']
        X = sm.add_constant(X)
        fit = sm.OLS(df_train_eqwt['sc1'], X).fit()

        # get the coefficient for the summed feature and the intercept
        coef = fit.params['sumfeature']
        const = fit.params['const']

        # now we need to assign this coefficient to all of the original
        # features and create a fake SKLL learner with these weights
        original_features = [c for c in df_train_eqwt.columns if c not in ['sc1',
                                                                           'sumfeature',
                                                                           'spkitemid']]
        coefs = pd.Series(dict([(origf, coef)
                                for origf in original_features] + [('const', const)]))
        df_coef = self.ols_coefficients_to_dataframe(coefs)

        # create fake SKLL learner with these coefficients
        learner = self.create_fake_skll_learner(df_coef)

        # we used all the features
        used_features = feature_columns

        return learner, fit, df_coef, used_features

    def train_rebalanced_lr(self, df_train, feature_columns):
        """
        Train `RebalancedLR` (formerly empWtBalanced) -
        Balanced empirical weights by changing betas
        [adapted from: https://stats.stackexchange.com/q/30876]

        Parameters
        ----------
        df_train : pd.DataFrame
            Data frame containing the features on which
            to train the model.
        feature_columns : list
            A list of feature columns to use in training the model.

        Returns
        -------
        learner : skll.Learner
            The SKLL learner object
        fit : statsmodels.RegressionResults
            A statsmodels regression results object.
        df_coef : pd.DataFrame
            The model coefficients in a data_frame
        used_features : list
            A list of features used in the final model.
        """
        # train a plain Linear Regression model
        X = df_train[feature_columns]
        X = sm.add_constant(X)
        fit = sm.OLS(df_train['sc1'], X).fit()

        # convert the model parameters into a data frame
        df_params = self.ols_coefficients_to_dataframe(fit.params)
        df_params = df_params.set_index('feature')

        # compute the betas for the non-intercept coefficients
        df_weights = df_params.loc[feature_columns]
        df_betas = df_weights.copy()

        df_train_std = df_train[feature_columns].std()
        df_betas['coefficient'] = (df_weights['coefficient'].multiply(df_train_std,
                                                                      axis='index') /
                                   df_train['sc1'].std())

        # replace each negative beta with delta and adjust
        # all the positive betas to account for this
        RT = 0.05
        df_positive_betas = df_betas[df_betas['coefficient'] > 0]
        df_negative_betas = df_betas[df_betas['coefficient'] < 0]
        delta = np.sum(df_positive_betas['coefficient']) * RT / len(df_negative_betas)
        df_betas['coefficient'] = df_betas.apply(lambda row: row['coefficient'] * (1 - RT)
                                                 if row['coefficient'] > 0 else delta, axis=1)

        # rescale the adjusted betas to get the new coefficients
        df_coef = df_betas['coefficient'] * df_train['sc1'].std()
        df_coef = df_coef.divide(df_train[feature_columns].std(), axis='index')

        # add the intercept back to the new coefficients
        df_coef['Intercept'] = df_params.loc['Intercept'].coefficient
        df_coef = df_coef.sort_index().reset_index()
        df_coef.columns = ['feature', 'coefficient']

        # create fake SKLL learner with these coefficients
        learner = self.create_fake_skll_learner(df_coef)

        # we used all the features
        used_features = feature_columns

        return learner, fit, df_coef, used_features

    def train_lasso_fixed_lambda_then_lr(self, df_train, feature_columns):
        """
        Train `LassoFixedLambdaThenLR` (formerly empWtLasso) -
        First do feature selection using lasso regression with
        a fixed lambda and then use only those features to train
        a second linear regression

        Parameters
        ----------
        df_train : pd.DataFrame
            Data frame containing the features on which
            to train the model.
        feature_columns : list
            A list of feature columns to use in training the model.

        Returns
        -------
        learner : skll.Learner
            The SKLL learner object
        fit : statsmodels.RegressionResults
            A statsmodels regression results object.
        df_coef : pd.DataFrame
            The model coefficients in a data_frame
        used_features : list
            A list of features used in the final model.
        """
        # train a Lasso Regression model with this featureset with a preset lambda
        p_lambda = sqrt(len(df_train) * log10(len(feature_columns)))

        # create a SKLL FeatureSet instance from the given data frame
        fs_train = FeatureSet.from_data_frame(df_train[feature_columns + ['sc1']],
                                              'train',
                                              labels_column='sc1')

        # note that 'alpha' in sklearn is different from this lambda
        # so we need to normalize looking at the sklearn objective equation
        p_alpha = p_lambda / len(df_train)
        l_lasso = Learner('Lasso', model_kwargs={'alpha': p_alpha, 'positive': True})
        l_lasso.train(fs_train, grid_search=False)

        # get the feature names that have the non-zero coefficients
        non_zero_features = list(l_lasso.model_params[0].keys())

        # now train a new vanilla linear regression with just the non-zero features
        X = df_train[non_zero_features]
        X = sm.add_constant(X)
        fit = sm.OLS(df_train['sc1'], X).fit()

        # get the coefficients data frame
        df_coef = self.ols_coefficients_to_dataframe(fit.params)

        # create fake SKLL learner with these coefficients
        learner = self.create_fake_skll_learner(df_coef)

        # we used only the non-zero features
        used_features = non_zero_features

        return learner, fit, df_coef, used_features

    def train_positive_lasso_cv_then_lr(self, df_train, feature_columns):
        """
        Train `PositiveLassoCVThenLR` (formerly empWtLassoBest) -
        First do feature selection using lasso regression optimized
        for log likelihood using cross validation and then use only
        those features to train a second linear regression

        Parameters
        ----------
        df_train : pd.DataFrame
            Data frame containing the features on which
            to train the model.
        feature_columns : list
            A list of feature columns to use in training the model.

        Returns
        -------
        learner : skll.Learner
            The SKLL learner object
        fit : statsmodels.RegressionResults
            A statsmodels regression results object.
        df_coef : pd.DataFrame
            The model coefficients in a data_frame
        used_features : list
            A list of features used in the final model.
        """
        # train a LassoCV outside of SKLL since it's not exposed there
        X = df_train[feature_columns].values
        y = df_train['sc1'].values
        clf = LassoCV(cv=10, positive=True, random_state=1234567890)
        model = clf.fit(X, y)

        # get the non-zero features from this model
        non_zero_features = []
        for feature, coefficient in zip(feature_columns, model.coef_):
            if coefficient != 0:
                non_zero_features.append(feature)

        # now train a new linear regression with just these non-zero features
        X = df_train[non_zero_features]
        X = sm.add_constant(X)
        fit = sm.OLS(df_train['sc1'], X).fit()

        # convert the model parameters into a data frame
        df_coef = self.ols_coefficients_to_dataframe(fit.params)

        # create fake SKLL learner with these coefficients
        learner = self.create_fake_skll_learner(df_coef)

        # we used only the non-zero features
        used_features = non_zero_features

        return learner, fit, df_coef, used_features

    def train_non_negative_lr(self, df_train, feature_columns):
        """
        Train `NNLR` (formerly empWtNNLS) -
        First do feature selection using non-negative least squares (NNLS)
        and then use only its non-zero features to train a regular linear regression.
        We do the regular LR at the end since we want an LR object so that we have access
        to R^2 and other useful statistics. There should be no difference
        between the non-zero coefficients from NNLS and the coefficients
        that end up coming out of the subsequent LR.

        Parameters
        ----------
        df_train : pd.DataFrame
            Data frame containing the features on which
            to train the model.
        feature_columns : list
            A list of feature columns to use in training the model.

        Returns
        -------
        learner : skll.Learner
            The SKLL learner object
        fit : statsmodels.RegressionResults
            A statsmodels regression results object.
        df_coef : pd.DataFrame
            The model coefficients in a data_frame
        used_features : list
            A list of features used in the final model.
        """
        # add an intercept to the features manually
        X = df_train[feature_columns].values
        intercepts = np.ones((len(df_train), 1))
        X_plus_intercept = np.concatenate([intercepts, X], axis=1)
        y = df_train['sc1'].values

        # fit an NNLS model on this data
        coefs, rnorm = nnls(X_plus_intercept, y)

        # check whether the intercept is set to 0 and if so then we need
        # to flip the sign and refit the model to ensure that it is always
        # kept in the model
        if coefs[0] == 0:
            intercepts = -1 * np.ones((len(df_train), 1))
            X_plus_intercept = np.concatenate([intercepts, X], axis=1)
            coefs, rnorm = nnls(X_plus_intercept, y)

        # separate the intercept and feature coefficients
        # intercept = coefs[0]
        coefficients = coefs[1:].tolist()

        # get the non-zero features from this model
        non_zero_features = []
        for feature, coefficient in zip(feature_columns, coefficients):
            if coefficient != 0:
                non_zero_features.append(feature)

        # now train a new linear regression with just these non-zero features
        X = df_train[non_zero_features]
        X = sm.add_constant(X)
        fit = sm.OLS(df_train['sc1'], X).fit()

        # convert this model's parameters to a data frame
        df_coef = self.ols_coefficients_to_dataframe(fit.params)

        # create fake SKLL learner with these coefficients
        learner = self.create_fake_skll_learner(df_coef)

        # we used only the non-zero features
        used_features = non_zero_features

        return learner, fit, df_coef, used_features

    def train_non_negative_lr_iterative(self, df_train, feature_columns):
        """
        Train `NNLR_iterative` -
        For applications where there is a concern that standard NNLS may not
        converge, an alternate method of training NNLR by iteratively fitting OLS
        models, checking the coefficients, and dropping negative coefficients.
        First, fit an OLS model. Then, identify any variables whose coefficients
        are negative. Drop these variables from the model. Finally, refit the
        model. If any coefficients are still negative, set these to zero.

        Parameters
        ----------
        df_train : pd.DataFrame
            Data frame containing the features on which
            to train the model.
        feature_columns : list
            A list of feature columns to use in training the model.

        Returns
        -------
        learner : skll.Learner
            The SKLL learner object.
        fit : statsmodels.RegressionResults
            A statsmodels regression results object.
        df_coef : pd.DataFrame
            The model coefficients in a data frame
        used_features : list
            A list of features used in the final model.
        """
        X = df_train[feature_columns]
        X = sm.add_constant(X)

        y = df_train['sc1']

        fit = sm.OLS(y, X).fit()

        positive_features = []
        for name, value in fit.params.items():
            if value >= 0 and name != 'const':
                positive_features.append(name)

        X = df_train[positive_features]
        X = sm.add_constant(X)

        fit = sm.OLS(y, X).fit()

        # if any parameters are still negative, set them to zero
        params = fit.params.copy()
        params = params.drop('const')
        if not (params >= 0).all():
            fit.params[(fit.params < 0) & (fit.params.index != 'const')] = 0

        # convert this model's parameters to a data frame
        df_coef = self.ols_coefficients_to_dataframe(fit.params)

        # create fake SKLL learner with these coefficients
        learner = self.create_fake_skll_learner(df_coef)

        # we used only the non-zero features
        used_features = positive_features

        return learner, fit, df_coef, used_features

    def train_lasso_fixed_lambda_then_non_negative_lr(self, df_train, feature_columns):
        """
        Train `LassoFixedLambdaThenNNLR` (formerly empWtDropNegLasso) -
        First do feature selection using lasso regression and positive only weights.
        Then fit an NNLR (see above) on those features.

        Parameters
        ----------
        df_train : pd.DataFrame
            Data frame containing the features on which
            to train the model.
        feature_columns : list
            A list of feature columns to use in training the model.

        Returns
        -------
        learner : skll.Learner
            The SKLL learner object
        fit : statsmodels.RegressionResults
            A statsmodels regression results object.
        df_coef : pd.DataFrame
            The model coefficients in a data_frame
        used_features : list
            A list of features used in the final model.
        """
        # train a Lasso Regression model with a preset lambda
        p_lambda = sqrt(len(df_train) * log10(len(feature_columns)))

        # create a SKLL FeatureSet instance from the given data frame
        fs_train = FeatureSet.from_data_frame(df_train[feature_columns + ['sc1']],
                                              'train',
                                              labels_column='sc1')

        # note that 'alpha' in sklearn is different from this lambda
        # so we need to normalize looking at the sklearn objective equation
        p_alpha = p_lambda / len(df_train)
        l_lasso = Learner('Lasso', model_kwargs={'alpha': p_alpha, 'positive': True})
        l_lasso.train(fs_train, grid_search=False)

        # get the feature names that have the non-zero coefficients
        non_zero_features = list(l_lasso.model_params[0].keys())

        # now train an NNLS regression using these non-zero features
        # first add an intercept to the features manually
        X = df_train[feature_columns].values
        intercepts = np.ones((len(df_train), 1))
        X_plus_intercept = np.concatenate([intercepts, X], axis=1)
        y = df_train['sc1'].values

        # fit an NNLS model on this data
        coefs, rnorm = nnls(X_plus_intercept, y)

        # check whether the intercept is set to 0 and if so then we need
        # to flip the sign and refit the model to ensure that it is always
        # kept in the model
        if coefs[0] == 0:
            intercepts = -1 * np.ones((len(df_train), 1))
            X_plus_intercept = np.concatenate([intercepts, X], axis=1)
            coefs, rnorm = nnls(X_plus_intercept, y)

        # separate the intercept and feature coefficients
        # even though we do not use intercept in the code
        # we define it here for readability
        # intercept = coefs[0]
        coefficients = coefs[1:].tolist()

        # get the non-zero features from this model
        non_zero_features = []
        for feature, coefficient in zip(feature_columns, coefficients):
            if coefficient != 0:
                non_zero_features.append(feature)

        # now train a new linear regression with just these non-zero features
        X = df_train[non_zero_features]
        X = sm.add_constant(X)
        fit = sm.OLS(df_train['sc1'], X).fit()

        # convert this model's parameters into a data frame
        df_coef = self.ols_coefficients_to_dataframe(fit.params)

        # create fake SKLL learner with these coefficients
        learner = self.create_fake_skll_learner(df_coef)

        # we used only the positive features
        used_features = non_zero_features

        return learner, fit, df_coef, used_features

    def train_lasso_fixed_lambda(self, df_train, feature_columns):
        """
        Train `LassoFixedLambda` (formerly lassoWtLasso) -
        A Lasso model with a fixed lambda

        Parameters
        ----------
        df_train : pd.DataFrame
            Data frame containing the features on which
            to train the model.
        feature_columns : list
            A list of feature columns to use in training the model.

        Returns
        -------
        learner : skll.Learner
            The SKLL learner object
        fit : statsmodels.RegressionResults
            A statsmodels regression results object or None.
        df_coef : pd.DataFrame
            The model coefficients in a data_frame
        used_features : list
            A list of features used in the final model.
        """
        # train a Lasso Regression model with a preset lambda
        p_lambda = sqrt(len(df_train) * log10(len(feature_columns)))

        # create a SKLL FeatureSet instance from the given data frame
        fs_train = FeatureSet.from_data_frame(df_train[feature_columns + ['sc1']],
                                              'train',
                                              labels_column='sc1')

        # note that 'alpha' in sklearn is different from this lambda
        # so we need to normalize looking at the sklearn objective equation
        alpha = p_lambda / len(df_train)
        learner = Learner('Lasso', model_kwargs={'alpha': alpha, 'positive': True})
        learner.train(fs_train, grid_search=False)

        # convert this model's parameters to a data frame
        df_coef = self.skll_learner_params_to_dataframe(learner)

        # there's no OLS fit object in this case
        fit = None

        # we used all the features
        used_features = feature_columns

        return learner, fit, df_coef, used_features

    def train_positive_lasso_cv(self, df_train, feature_columns):
        """
        Train `PositiveLassoCV` (formerly lassoWtLassoBest) -
        Feature selection using lasso regression optimized for log likelihood
        using cross validation.


        Parameters
        ----------
        df_train : pd.DataFrame
            Data frame containing the features on which
            to train the model.
        feature_columns : list
            A list of feature columns to use in training the model.

        Returns
        -------
        learner : skll.Learner
            The SKLL learner object
        fit : statsmodels.RegressionResults
            A statsmodels regression results object or None.
        df_coef : pd.DataFrame
            The model coefficients in a data_frame
        used_features : list
            A list of features used in the final model.
        """
        # train a LassoCV outside of SKLL since it's not exposed there
        X = df_train[feature_columns].values
        y = df_train['sc1'].values
        clf = LassoCV(cv=10, positive=True, random_state=1234567890)
        model = clf.fit(X, y)

        # save the non-zero model coefficients and intercept to a data frame
        non_zero_features, non_zero_feature_values = [], []
        for feature, coefficient in zip(feature_columns, model.coef_):
            if coefficient != 0:
                non_zero_features.append(feature)
                non_zero_feature_values.append(coefficient)

        # initialize the coefficient data frame with just the intercept
        df_coef = pd.DataFrame([('Intercept', model.intercept_)])
        df_coef = df_coef.append(list(zip(non_zero_features,
                                          non_zero_feature_values)), ignore_index=True)
        df_coef.columns = ['feature', 'coefficient']

        # create a fake SKLL learner with these non-zero weights
        learner = self.create_fake_skll_learner(df_coef)

        # there's no OLS fit object in this case
        fit = None

        # we used only the non-zero features
        used_features = non_zero_features

        return learner, fit, df_coef, used_features

    def train_score_weighted_lr(self, df_train, feature_columns):
        """
        Train `ScoreWeightedLR` -
        Linear regression model weighted by score.

        Parameters
        ----------
        df_train : pd.DataFrame
            Data frame containing the features on which
            to train the model.
        feature_columns : list
            A list of feature columns to use in training the model.

        Returns
        -------
        learner : skll.Learner
            The SKLL learner object
        fit : statsmodels.RegressionResults
            A statsmodels regression results object or None.
        df_coef : pd.DataFrame
            The model coefficients in a data_frame
        used_features : list
            A list of features used in the final model.
        """
        # train weighted least squares regression
        # get the feature columns

        X = df_train[feature_columns]

        # add the intercept
        X = sm.add_constant(X)

        # define the weights as inverse proportion of total
        # number of data points for each score
        score_level_dict = df_train['sc1'].value_counts()
        expected_proportion = 1 / len(score_level_dict)
        score_weights_dict = {sc1: expected_proportion / count
                              for sc1, count in score_level_dict.items()}
        weights = [score_weights_dict[sc1] for sc1 in df_train['sc1']]

        # fit the model
        fit = sm.WLS(df_train['sc1'], X, weights=weights).fit()
        df_coef = self.ols_coefficients_to_dataframe(fit.params)
        learner = self.create_fake_skll_learner(df_coef)

        # we used all the features
        used_features = feature_columns

        return learner, fit, df_coef, used_features

    def train_builtin_model(self,
                            model_name,
                            df_train,
                            experiment_id,
                            filedir,
                            figdir,
                            file_format='csv'):
        """
        Train one of the :ref:`built-in linear regression models <builtin_models>`.

        Parameters
        ----------
        model_name : str
            Name of the built-in model to train.
        df_train : pandas DataFrame
            Data frame containing the features on which
            to train the model. The data frame must contain the ID column named
            `spkitemid` and the numeric label column named `sc1`.
        experiment_id : str
            The experiment ID.
        filedir : str
            Path to the `output` experiment output directory.
        figdir : str
            Path to the `figure` experiment output directory.
        file_format : {'csv', 'tsv', 'xlsx'}, optional
            The format in which to save files.
            Defaults to 'csv'.

        Returns
        -------
        learner : `Learner` object
            SKLL `LinearRegression` `Learner <https://skll.readthedocs.io/en/latest/api/learner.html#skll.learner.Learner>`_ object containing
            the coefficients learned by training the built-in model.
        """
        # get the columns that actually contain the feature values
        feature_columns = [c for c in df_train.columns if c not in ['spkitemid', 'sc1']]

        # LinearRegression
        if model_name == 'LinearRegression':
            result = self.train_linear_regression(df_train, feature_columns)

        # EqualWeightsLR
        elif model_name == 'EqualWeightsLR':
            result = self.train_equal_weights_lr(df_train, feature_columns)

        # RebalancedLR
        elif model_name == 'RebalancedLR':
            result = self.train_rebalanced_lr(df_train, feature_columns)

        # LassoFixedLambdaThenLR
        elif model_name == 'LassoFixedLambdaThenLR':
            result = self.train_lasso_fixed_lambda_then_lr(df_train, feature_columns)

        # PositiveLassoCVThenLR
        elif model_name == 'PositiveLassoCVThenLR':
            result = self.train_positive_lasso_cv_then_lr(df_train, feature_columns)

        # NNLR
        elif model_name == 'NNLR':
            result = self.train_non_negative_lr(df_train, feature_columns)

        # NNLRIterative
        elif model_name == 'NNLRIterative':
            result = self.train_non_negative_lr_iterative(df_train, feature_columns)

        # LassoFixedLambdaThenNNLR
        elif model_name == 'LassoFixedLambdaThenNNLR':
            result = self.train_lasso_fixed_lambda_then_non_negative_lr(df_train, feature_columns)

        # LassoFixedLambda
        elif model_name == 'LassoFixedLambda':
            result = self.train_lasso_fixed_lambda(df_train, feature_columns)

        # PositiveLassoCV
        elif model_name == 'PositiveLassoCV':
            result = self.train_positive_lasso_cv(df_train, feature_columns)

        # ScoreWeightedLR
        elif model_name == 'ScoreWeightedLR':
            result = self.train_score_weighted_lr(df_train, feature_columns)

        writer = DataWriter(experiment_id)
        frames = []

        # unpack all results
        learner, fit, df_coef, used_features = result

        # add raw coefficients to frame list
        frames.append({'name': 'coefficients', 'frame': df_coef})

        # compute the standardized and relative coefficients (betas) for the
        # non-intercept features and save to a file
        df_betas = df_coef.set_index('feature').loc[used_features]
        df_betas = df_betas.multiply(df_train[used_features].std(),
                                     axis='index') / df_train['sc1'].std()
        df_betas.columns = ['standardized']
        df_betas['relative'] = df_betas / sum(abs(df_betas['standardized']))
        df_betas.reset_index(inplace=True)

        # add betas to frame list
        frames.append({'name': 'betas', 'frame': df_betas})

        # save the OLS fit object and its summary to files
        if fit:
            ols_file = join(filedir, '{}.ols'.format(experiment_id))
            summary_file = join(filedir, '{}_ols_summary.txt'.format(experiment_id))
            with open(ols_file, 'wb') as olsf, open(summary_file, 'w') as summf:
                pickle.dump(fit, olsf)
                summf.write(str(fit.summary()))

            # create a data frame with main model fit metrics and save to the file
            df_model_fit = self.model_fit_to_dataframe(fit)

            # add model_fit to frame list
            frames.append({'name': 'model_fit', 'frame': df_model_fit})

        # save the SKLL model to a file
        model_file = join(filedir, '{}.model'.format(experiment_id))
        learner.save(model_file)

        container = DataContainer(frames)
        writer.write_experiment_output(filedir, container, file_format=file_format)

        self.learner = learner

        return learner

    def train_skll_model(self,
                         model_name,
                         df_train,
                         experiment_id,
                         filedir,
                         figdir,
                         file_format='csv',
                         custom_fixed_parameters=None,
                         custom_objective=None,
                         predict_expected_scores=False):
        """
        Train a SKLL classification or regression model.

        Parameters
        ----------
        model_name : str
            Name of the SKLL model to train.
        df_train : pandas DataFrame
            Data frame containing the features on which
            to train the model.
        experiment_id : str
            The experiment ID.
        filedir : str
            Path to the `output` experiment output directory.
        figdir : str
            Path to the `figure` experiment output directory.
        file_format : {'csv', 'tsv', 'xlsx'}, optional
            The format in which to save files. For SKLL models,
            this argument does not actually change the format of
            the output files at this time, as no betas are computed.
            Defaults to 'csv'.
        custom_fixed_parameters : dict, optional
            A dictionary containing any fixed parameters for the SKLL
            model.
            Defaults to ``None``.
        custom_objective : str, optional
            Name of custom user-specified objective. If not specified
            or `None`, `neg_mean_squared_error` is used as the objective.
            Defaults to `None`.
        predict_expected_scores : bool, optional
            Whether we want the trained classifiers to predict expected scores.
            Defaults to `False`.

        Returns
        -------
        Tuple containing a SKLL Learner object of the appropriate type
        and the chosen tuning objective.
        """
        # Instantiate the given SKLL learner and set its probability value
        # and fixed parameters appropriately
        model_kwargs = custom_fixed_parameters if custom_fixed_parameters is not None else {}
        learner = Learner(model_name,
                          model_kwargs=model_kwargs,
                          probability=predict_expected_scores)

        # get the features, IDs, and labels from the given data frame
        feature_columns = [c for c in df_train.columns if c not in ['spkitemid', 'sc1']]
        features = df_train[feature_columns].to_dict(orient='records')
        ids = df_train['spkitemid'].tolist()
        labels = df_train['sc1'].tolist()

        # create a FeatureSet and train the model
        fs = FeatureSet('train', ids=ids, labels=labels, features=features)

        # If we are training a SKLL regressor, then we want to use either the
        # user-specified objective or `neg_mean_squared_error`. If it's SKLL
        # classifier, then the choice is between the user-specified objective
        # and `f1_score_micro`.
        if learner.model_type._estimator_type == 'regressor':
            objective = 'neg_mean_squared_error' if not custom_objective else custom_objective
        else:
            objective = 'f1_score_micro' if not custom_objective else custom_objective

        learner.train(fs, grid_search=True, grid_objective=objective, grid_jobs=1)

        # TODO: compute betas for linear SKLL models?

        # save the SKLL model to disk with the given model name prefix
        model_file = join(filedir, '{}.model'.format(experiment_id))
        learner.save(model_file)

        self.learner = learner

        # return the SKLL learner object and the chosen objective
        return learner, objective

    def train(self,
              configuration,
              data_container,
              filedir,
              figdir,
              file_format='csv'):
        """
        The main driver function to train the given model on the given
        data and save the results in the given directories using the
        given experiment ID as the prefix.

        parameters
        ----------
        configuration : configuration_parser.Configuration
            A configuration object containing `experiment_id` and `model_name`
        data_container : container.DataContainer
            A data_container object containing `train_preprocessed_features`
        filedir : str
            Path to the `output` experiment output directory.
        figdir : str
            Path to the `figure` experiment output directory.
        file_format : {'csv', 'tsv', 'xlsx'}, optional
            The format in which to save files.
            Defaults to 'csv'.

        Returns
        -------
        name : SKLL Learner object
        """

        Analyzer.check_param_names(configuration, ['model_name', 'experiment_id'])
        Analyzer.check_frame_names(data_container, ['train_preprocessed_features'])

        model_name = configuration['model_name']
        experiment_id = configuration['experiment_id']

        df_train = data_container['train_preprocessed_features']

        args = [model_name, df_train, experiment_id, filedir, figdir]
        kwargs = {'file_format': file_format}

        # add user-specified SKLL objective to the arguments if we are
        # training a SKLL model
        if is_skll_model(model_name):
            kwargs.update({'custom_fixed_parameters': configuration['skll_fixed_parameters'],
                           'custom_objective': configuration['skll_objective'],
                           'predict_expected_scores': configuration['predict_expected_scores']})
            model, chosen_objective = self.train_skll_model(*args, **kwargs)
            configuration['skll_objective'] = chosen_objective
        else:
            model = self.train_builtin_model(*args, **kwargs)

        return model

    def predict(self, df, min_score, max_score, predict_expected=False):
        """
        Get the raw predictions of the given SKLL model on the data
        contained in the given data frame.

        Parameters
        ----------
        df : pandas DataFrame
            Data frame containing features on which to make the predictions.
            The data must contain pre-processed feature values, an ID column
            named `spkitemid`, and a label column named `sc1`.
        min_score : int
            Minimum score level to be used if computing expected scores.
        max_score : int
            Maximum score level to be used if computing expected scores.
        predict_expected : bool, optional
            Predict expected scores for classifiers that return probability
            distributions over score. This will be ignored with a warning
            if the specified model does not support probability distributions.
            Note also that this assumes that the score range consists of
            contiguous integers - starting at `min_score` and ending at
            `max_score`. Defaults to `False`.

        Returns
        -------
        df_predictions : pandas DataFrame
            Data frame containing the raw predictions, the IDs, and the
            human scores.

        Raises
        ------
        ValueError
            If the model cannot predict probability distributions and
            `predict_expected` is set to `True` or if the score range
            specified by `min_score` and `max_score` does not match
            what the model predicts in its probability distribution.
        """
        model = self.learner

        feature_columns = [c for c in df.columns if c not in ['spkitemid', 'sc1']]
        features = df[feature_columns].to_dict(orient='records')
        ids = df['spkitemid'].tolist()

        # if we have the labels, save them in the featureset
        labels = None
        if 'sc1' in df:
            labels = df['sc1'].tolist()

        fs = FeatureSet('data', ids=ids, labels=labels, features=features)
        # if we are predicting expected scores, then call a different function
        predictions = compute_expected_scores_from_model(model,
                                                         fs,
                                                         min_score,
                                                         max_score) if predict_expected else model.predict(fs)

        df_predictions = pd.DataFrame()
        df_predictions['spkitemid'] = ids
        df_predictions['raw'] = predictions

        # save the labels in the dataframe if they existed in the first place
        if labels:
            df_predictions['sc1'] = labels

        return df_predictions

    def predict_train_and_test(self,
                               df_train,
                               df_test,
                               configuration):
        """
        Generate raw, scaled, and trimmed predictions of `model`
        on the given training and testing data.

        Parameters
        ----------
        df_train : pandas DataFrame
            Data frame containing the pre-processed training
            set features.
        df_test : pandas DataFrame
            Data frame containing the pre-processed test
            set features.
        configuration : configuration_parser.Configuration
            A configuration object containing `trim_max` and `trim_min`

        Returns
        -------
        List of data frames containing predictions and other
        information.
        """

        Analyzer.check_param_names(configuration, ['trim_max',
                                                   'trim_min',
                                                   'trim_tolerance'])

        (trim_min,
         trim_max,
         trim_tolerance) = configuration.get_trim_min_max_tolerance()

        predict_expected_scores = configuration['predict_expected_scores']

        df_train_predictions = self.predict(df_train,
                                            int(trim_min),
                                            int(trim_max),
                                            predict_expected=predict_expected_scores)
        df_test_predictions = self.predict(df_test,
                                           int(trim_min),
                                           int(trim_max),
                                           predict_expected=predict_expected_scores)

        # get the mean and SD of the training set predictions
        train_predictions_mean = df_train_predictions['raw'].mean()
        train_predictions_sd = df_train_predictions['raw'].std()

        # get the mean and SD of the human labels
        human_labels_mean = df_train['sc1'].mean()
        human_labels_sd = df_train['sc1'].std()

        logging.info('Processing train set predictions.')
        df_train_predictions = FeaturePreprocessor.process_predictions(df_train_predictions,
                                                                       train_predictions_mean,
                                                                       train_predictions_sd,
                                                                       human_labels_mean,
                                                                       human_labels_sd,
                                                                       trim_min,
                                                                       trim_max,
                                                                       trim_tolerance)

        logging.info('Processing test set predictions.')
        df_test_predictions = FeaturePreprocessor.process_predictions(df_test_predictions,
                                                                      train_predictions_mean,
                                                                      train_predictions_sd,
                                                                      human_labels_mean,
                                                                      human_labels_sd,
                                                                      trim_min,
                                                                      trim_max,
                                                                      trim_tolerance)

        df_postproc_params = pd.DataFrame([{'trim_min': trim_min,
                                            'trim_max': trim_max,
                                            'trim_tolerance': trim_tolerance,
                                            'h1_mean': human_labels_mean,
                                            'h1_sd': human_labels_sd,
                                            'train_predictions_mean': train_predictions_mean,
                                            'train_predictions_sd': train_predictions_sd}])

        datasets = [{'name': 'pred_train', 'frame': df_train_predictions},
                    {'name': 'pred_test', 'frame': df_test_predictions},
                    {'name': 'postprocessing_params', 'frame': df_postproc_params}]

        # configuration options that are entirely for internal use
        internal_options_dict = {'train_predictions_mean': train_predictions_mean,
                                 'train_predictions_sd': train_predictions_sd,
                                 'human_labels_mean': human_labels_mean,
                                 'human_labels_sd': human_labels_sd}

        new_configuration = configuration.copy()
        for key, value in internal_options_dict.items():
            new_configuration[key] = value

        return new_configuration, DataContainer(datasets=datasets)

    def get_feature_names(self):
        """
        Get the feature names, if available.

        Returns
        -------
        feature_names : list or None
            A list of feature names, or None if no learner was trained.
        """
        if self.learner is not None:
            return self.learner.feat_vectorizer.get_feature_names()
        return None

    def get_intercept(self):
        """
        Get the intercept of the model, if available.

        Returns
        -------
        intercept : float or None
           The intercept of the model.
        """
        if self.learner is not None:
            return self.learner.model.intercept_
        return None

    def get_coefficients(self):
        """
        Get the coefficients of the model, if available.

        Returns
        -------
        coefficients : np.array or None
           The coefficients of the model.
        """
        if self.learner is not None:
            return self.learner.model.coef_
        return None

    def scale_coefficients(self,
                           configuration):
        """
        Scale coefficients and intercept using human scores and model
        prediction on the training set. This procedure approximates
        what is done in operational setting but does not apply
        trimming to predictions.

        Parameters
        ----------
        configuration : configuration_parser.Configuration
            A configuration object containing `train_predictions_mean`,
            and `train_predictions_sd`, and `human_labels_sd`.

        Returns
        -------
        data_container : container.DataContainer
            A data_container object containing `coefficients_scaled`
            This DataFrame contains the scaled coefficients
            and the feature names, along with the intercept.

        Raises
        ------
        RuntimeError
            If the model is non-linear and no coefficients are available.
        """

        Analyzer.check_param_names(configuration, ['train_predictions_mean',
                                                   'train_predictions_sd',
                                                   'human_labels_sd'])

        train_predictions_mean = configuration['train_predictions_mean']
        train_predictions_sd = configuration['train_predictions_sd']
        h1_sd = configuration['human_labels_sd']

        feature_names = self.get_feature_names()

        # try to get the model coefficients, if available
        try:
            coefficients = self.get_coefficients()
        except AttributeError:
            raise RuntimeError("no coefficients available for this model.")

        intercept = self.get_intercept()

        # scale the coefficients and the intercept
        scaled_coefficients = coefficients * h1_sd / train_predictions_sd

        # adjust the intercept to set the mean predicted score
        # to the mean of the training variable
        new_intercept = intercept * (h1_sd / train_predictions_sd)
        new_intercept += train_predictions_mean * (1 - h1_sd / train_predictions_sd)

        intercept_and_feature_names = ['Intercept'] + feature_names
        intercept_and_feature_values = [new_intercept] + list(scaled_coefficients)

        # create a data frame with new values
        df_scaled_coefficients = pd.DataFrame({'feature': intercept_and_feature_names,
                                               'coefficient': intercept_and_feature_values},
                                              columns=['feature', 'coefficient'])

        scaled_dataset = [{'name': 'coefficients_scaled',
                           'frame': df_scaled_coefficients}]

        return DataContainer(datasets=scaled_dataset)
