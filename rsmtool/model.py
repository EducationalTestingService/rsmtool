"""
Functions dealing with training built-in or SKLL models

:author: Nitin Madnani (nmadnani@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
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

builtin_models = ['LinearRegression',
                  'EqualWeightsLR',
                  'ScoreWeightedLR',
                  'RebalancedLR',
                  'NNLR',
                  'LassoFixedLambdaThenNNLR',
                  'LassoFixedLambdaThenLR',
                  'PositiveLassoCVThenLR',
                  'LassoFixedLambda',
                  'PositiveLassoCV']

skll_models = ['AdaBoostRegressor',
               'DecisionTreeRegressor',
               'ElasticNet',
               'GradientBoostingRegressor',
               'KNeighborsRegressor',
               'Lasso',
               'LinearSVR',
               'RandomForestRegressor',
               'Ridge',
               'SGDRegressor',
               'SVR']


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
    df_non_intercept = pd.DataFrame(coefs.filter(non_intercept_columns), columns=['coefficient'])
    df_non_intercept.index.name = 'feature'
    df_non_intercept = df_non_intercept.sort_index()
    df_non_intercept.reset_index(inplace=True)

    # now create a data frame that just has the intercept
    df_intercept = pd.DataFrame([{'feature': 'Intercept',
                                  'coefficient': coefs['const']}])

    # append the non-intercept frame to the intercept one
    df_coef = df_intercept.append(df_non_intercept, ignore_index=True)

    # we always want to have the feature column first
    df_coef = df_coef[['feature', 'coefficient']]

    return df_coef


def skll_learner_params_to_dataframe(learner):
    """
    Take the given SKLL learner object and return a data
    frame containing its parameters.

    Parameters
    ----------
    learner : skll Learner object

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
    df_coef = df_intercept.append(df_non_intercept, ignore_index=True)

    # we always want to have the feature column first
    df_coef = df_coef[['feature', 'coefficient']]

    return df_coef


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

    # get the logger
    logger = logging.getLogger(__name__)

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
                logger.warning("No coefficient was estimated for "
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


def train_builtin_model(model_name, df_train, experiment_id, csvdir, figdir):
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
    csvdir : str
        Path to the `output` experiment output directory.
    figdir : str
        Path to the `figure` experiment output directory.

    Returns
    -------
    learner : `Learner` object
        SKLL `LinearRegression` `Learner <http://skll.readthedocs.io/en/latest/api/skll.html#skll.Learner>`_ object containing the
        coefficients learned by training the built-in model.
    """
    # get the columns that actually contain the feature values
    feature_columns = [c for c in df_train.columns if c not in ['spkitemid', 'sc1']]

    # LinearRegression (formerly empWt) : simple linear regression
    if model_name == 'LinearRegression':

        # get the feature columns
        X = df_train[feature_columns]

        # add the intercept
        X = sm.add_constant(X)

        # fit the model
        fit = sm.OLS(df_train['sc1'], X).fit()
        df_coef = ols_coefficients_to_dataframe(fit.params)
        learner = create_fake_skll_learner(df_coef)

        # we used all the features
        used_features = feature_columns

    # EqualWeightsLR (formerly eqWt) : all features get equal weight
    elif model_name == 'EqualWeightsLR':
        # we first compute a single feature that is simply the sum of all features
        df_train_eqwt = df_train.copy()
        df_train_eqwt['sumfeature'] = df_train_eqwt[feature_columns].apply(lambda row: np.sum(row), axis=1)

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
        coefs = pd.Series(dict([(origf, coef) for origf in original_features] + [('const', const)]))
        df_coef = ols_coefficients_to_dataframe(coefs)

        # create fake SKLL learner with these coefficients
        learner = create_fake_skll_learner(df_coef)

        # we used all the features
        used_features = feature_columns

    # RebalancedLR (formerly empWtBalanced) : balanced empirical weights
    # by changing betas [adapted from http://bit.ly/UTP7gS]
    elif model_name == 'RebalancedLR':

        # train a plain Linear Regression model
        X = df_train[feature_columns]
        X = sm.add_constant(X)
        fit = sm.OLS(df_train['sc1'], X).fit()

        # convert the model parameters into a data frame
        df_params = ols_coefficients_to_dataframe(fit.params)
        df_params = df_params.set_index('feature')

        # compute the betas for the non-intercept coefficients
        df_weights = df_params.loc[feature_columns]
        df_betas = df_weights.copy()
        df_betas['coefficient'] = df_weights['coefficient'].multiply(df_train[feature_columns].std(), axis='index') / df_train['sc1'].std()

        # replace each negative beta with delta and adjust
        # all the positive betas to account for this
        RT = 0.05
        df_positive_betas = df_betas[df_betas['coefficient'] > 0]
        df_negative_betas = df_betas[df_betas['coefficient'] < 0]
        delta = np.sum(df_positive_betas['coefficient']) * RT / len(df_negative_betas)
        df_betas['coefficient'] = df_betas.apply(lambda row: row['coefficient'] * (1-RT) if row['coefficient'] > 0 else delta, axis=1)

        # rescale the adjusted betas to get the new coefficients
        df_coef = (df_betas['coefficient'] * df_train['sc1'].std()).divide(df_train[feature_columns].std(), axis='index')

        # add the intercept back to the new coefficients
        df_coef['Intercept'] = df_params.loc['Intercept'].coefficient
        df_coef = df_coef.sort_index().reset_index()
        df_coef.columns = ['feature', 'coefficient']

        # create fake SKLL learner with these coefficients
        learner = create_fake_skll_learner(df_coef)

        # we used all the features
        used_features = feature_columns

    # LassoFixedLambdaThenLR (formerly empWtLasso) : First do feature
    # selection using lasso regression with a fixed lambda and then
    # use only those features to train a second linear regression
    elif model_name == 'LassoFixedLambdaThenLR':

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
        df_coef = ols_coefficients_to_dataframe(fit.params)

        # create fake SKLL learner with these coefficients
        learner = create_fake_skll_learner(df_coef)

        # we used only the non-zero features
        used_features = non_zero_features

    # PositiveLassoCVThenLR (formerly empWtLassoBest) : First do feature
    # selection using lasso regression optimized for log likelihood using
    # cross validation and then use only those features to train a
    # second linear regression
    elif model_name == 'PositiveLassoCVThenLR':

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
        df_coef = ols_coefficients_to_dataframe(fit.params)

        # create fake SKLL learner with these coefficients
        learner = create_fake_skll_learner(df_coef)

        # we used only the non-zero features
        used_features = non_zero_features

    # NNLR (formerly empWtNNLS) : First do feature selection using
    # non-negative least squares (NNLS) and then use only its non-zero
    # features to train a regular linear regression. We do the regular
    # LR at the end since we want an LR object so that we have access
    # to R^2 and other useful statistics. There should be no difference
    # between the non-zero coefficients from NNLS and the coefficients
    # that end up coming out of the subsequent LR.
    elif model_name == 'NNLR':

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
        intercept = coefs[0]
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
        df_coef = ols_coefficients_to_dataframe(fit.params)

        # create fake SKLL learner with these coefficients
        learner = create_fake_skll_learner(df_coef)

        # we used only the non-zero features
        used_features = non_zero_features

    # LassoFixedLambdaThenNNLR (formerly empWtDropNegLasso): First do
    # feature selection using lasso regression and positive only weights.
    # Then fit an NNLR (see above) on those features.
    elif model_name == 'LassoFixedLambdaThenNNLR':

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
        intercept = coefs[0]
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
        df_coef = ols_coefficients_to_dataframe(fit.params)

        # create fake SKLL learner with these coefficients
        learner = create_fake_skll_learner(df_coef)

        # we used only the positive features
        used_features = non_zero_features

    # LassoFixedLambda (formerly lassoWtLasso) : Lasso model with
    # a fixed lambda
    elif model_name == 'LassoFixedLambda':

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
        df_coef = skll_learner_params_to_dataframe(learner)

        # there's no OLS fit object in this case
        fit = None

        # we used all the features
        used_features = feature_columns

    # PositiveLassoCV (formerly lassoWtLassoBest) : feature selection
    # using lasso regression optimized for log likelihood using cross
    # validation.
    elif model_name == 'PositiveLassoCV':

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
        learner = create_fake_skll_learner(df_coef)

        # there's no OLS fit object in this case
        fit = None

        # we used only the non-zero features
        used_features = non_zero_features

    elif model_name == 'ScoreWeightedLR':

        # train weighted least squares regression
        # get the feature columns

        X = df_train[feature_columns]

        # add the intercept
        X = sm.add_constant(X)

        # define the weights as inverse proportion of total number of data points for each score
        score_level_dict = df_train['sc1'].value_counts()
        expected_proportion = 1/len(score_level_dict)
        score_weights_dict = {sc1: expected_proportion/count for sc1, count in score_level_dict.items()}
        weights = [score_weights_dict[sc1] for sc1 in df_train['sc1']]

        # fit the model
        fit = sm.WLS(df_train['sc1'], X, weights=weights).fit()
        df_coef = ols_coefficients_to_dataframe(fit.params)
        learner = create_fake_skll_learner(df_coef)

        # we used all the features
        used_features = feature_columns


    # save the raw coefficients to a file
    df_coef.to_csv(join(csvdir, '{}_coefficients.csv'.format(experiment_id)), index=False)

    # compute the standardized and relative coefficients (betas) for the
    # non-intercept features and save to a file
    df_betas = df_coef.set_index('feature').loc[used_features]
    df_betas = df_betas.multiply(df_train[used_features].std(), axis='index') / df_train['sc1'].std()
    df_betas.columns = ['standardized']
    df_betas['relative'] = df_betas / sum(abs(df_betas['standardized']))
    df_betas.reset_index(inplace=True)
    df_betas.to_csv(join(csvdir, '{}_betas.csv'.format(experiment_id)), index=False)

    # save the OLS fit object and its summary to files
    if fit:
        ols_file = join(csvdir, '{}.ols'.format(experiment_id))
        summary_file = join(csvdir, '{}_ols_summary.txt'.format(experiment_id))
        with open(ols_file, 'wb') as olsf, open(summary_file, 'w') as summf:
            pickle.dump(fit, olsf)
            summf.write(str(fit.summary()))

        # create a data frame with main model fit metrics and save to the file
        df_model_fit = model_fit_to_dataframe(fit)
        model_fit_file = join(csvdir, '{}_model_fit.csv'.format(experiment_id))
        df_model_fit.to_csv(model_fit_file, index=False)

    # save the SKLL model to a file
    model_file = join(csvdir, '{}.model'.format(experiment_id))
    learner.save(model_file)

    return learner


def train_skll_model(model_name, df_train, experiment_id, csvdir, figdir):
    """
    Train a SKLL regression model.

    Parameters
    ----------
    model_name : str
        Name of the SKLL model to train.
    df_train : pandas DataFrame
        Data frame containing the features on which
        to train the model.
    experiment_id : str
        The experiment ID.
    csvdir : str
        Path to the `output` experiment output directory.
    figdir : str
        Path to the `figure` experiment output directory.

    Returns
    -------
    learner : skll Learner object
        SKLL Learner object of the appropriate type.
    """
    # instantiate the given SKLL learner
    learner = Learner(model_name)

    # get the features, IDs, and labels from the given data frame
    feature_columns = [c for c in df_train.columns if c not in ['spkitemid', 'sc1']]
    features = df_train[feature_columns].to_dict(orient='records')
    ids = df_train['spkitemid'].tolist()
    labels = df_train['sc1'].tolist()

    # create a FeatureSet and train the model
    fs = FeatureSet('train', ids=ids, labels=labels, features=features)

    # if it's a regression model, then our grid objective should be
    # pearson and otherwise it should be accuracy
    if model_name in skll_models:
        objective = 'pearson'
    else:
        objective = 'f1_score_micro'

    learner.train(fs, grid_search=True, grid_objective=objective, grid_jobs=1)

    # TODO: compute betas for linear SKLL models?

    # save the SKLL model to disk with the given model name prefix
    model_file = join(csvdir, '{}.model'.format(experiment_id))
    learner.save(model_file)

    # return the SKLL learner object
    return learner


def train_model(model_name, df_train, experiment_id, csvdir, figdir):
    """
    The main driver function to train the given model on the given
    data and save the results in the given directories using the
    given experiment ID as the prefix.

    parameters
    ----------
    model_name : str
        Name of the model to train.
    df_train : pandas DataFrame
        Data frame containing the features on which
        to train the model.
    experiment_id : str
        The experiment ID.
    csvdir : str
        Path to the `output` experiment output directory.
    figdir : str
        Path to the `figure` experiment output directory.

    Returns
    -------
    name : SKLL Learner object
    """
    call_args = [model_name, df_train, experiment_id, csvdir, figdir]
    model = train_builtin_model(*call_args) if model_name in builtin_models \
        else train_skll_model(*call_args)
    return model


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

    if model_name in builtin_models:
        model_type = 'BUILTIN'
    elif model_name in skll_models:
            model_type = 'SKLL'
    else:
        raise ValueError("The specified model {} "
                         "was not found. Please "
                         "check the spelling.".format(model_name))

    return model_type
