.. _builtin_models:

Built-in RSMTool Linear Regression Models
-----------------------------------------

Models which use the full feature set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``LinearRegression``: A model that learns empirical regression weights using ordinary least squares regression (OLS).

- ``EqualWeightsLR``:  A model with all feature weights set to 1.0; a naive model.

- ``ScoreWeightedLR``: a model that learns empirical regression weights using weighted least sqaures. The weights are determined based on the number of responses with different score levels. Score levels with lower number of responses are assigned higher weight.

- ``RebalancedLR`` -  empirical regression weights are rebalanced by using a small portion of positive weights to replace negative beta values. This model has no negative coefficients.


.. _automatic_feature_selection_models:

Models with automatic feature selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``LassoFixedLambdaThenLR``: A model that learns empirical OLS regression weights with feature selection using Lasso regression with all coefficients set to positive. The hyperparameter ``lambda`` is set to ``sqrt(n-lg(p))`` where ``n`` is the number of responses and ``p`` is the number of features. This approach was chosen to balance the penalties for error vs. penalty for two many coefficients to force Lasso perform more aggressive feature selection, so it may not necessarily achieve the best possible performance. The feature set selected by LASSO is then used to fit an OLS linear regression. Note that while the original Lasso model is constrained to positive coefficients only, small negative coefficients may appear when the coefficients are re-estimated using OLS regression.

- ``PositiveLassoCVThenLR``: A model that learns empirical OLS regression weights with feature selection using Lasso regression with all coefficients set to positive. The hyperparameter ``lambda`` is optimized using crossvalidation for loglikehood. The feature set selected by LASSO is then used to fit an OLS linear regression. Note that this approach will likely produce a model with a large N features and any advantages of running Lasso would be effectively negated by latter adding those features to OLS regression.

- ``NNLR``: A model that learns empirical OLS regression weights with feature selection using non-negative least squares regression. Note that only the coefficients are constrained to be positive: the intercept can be either positive or negative.

- ``NNLRIterative``: A model that learns empirical OLS regression weights with feature selection using an iterative implementation of non-negative least squares regression. Under this implementation, an initial OLS model is fit. Then, any variables whose coefficients are negative are dropped and the model is re-fit. Any coefficients that are still negative after re-fitting are set to zero.

- ``LassoFixedLambdaThenNNLR``: A model that learns empirical OLS regression weights with feature selection using Lasso regression as above followed by non-negative least squares regression. The latter ensures that no feature has negative coefficients even when the coefficients are estimated using least squares without penalization.

- ``LassoFixedLambda``: same as ``LassoFixedLambdaThenLR`` but the model uses the original Lasso weights. Note that the coefficients in Lasso model are estimated using an optimization routine which may produce slightly different results on different systems.

- ``PositiveLassoCV``: same as `PositiveLassoCVThenLR` but using the original Lasso weights. Please note: the coefficients in Lasso model are estimated using an optimization routine which may produce slightly different results on different systems.

.. note::

    1. ``NNLR``, ``NNLRIterative``, ``LassoFixedLambdaThenNNLR``, ``LassoFixedLambda`` and ``PositiveLassoCV`` all have no negative coefficients.

    2. For all feature selection models, the final set of features will be saved in the ``feature`` folder in the experiment output directory.
