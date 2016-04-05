# List of available machine learning models

## Linear models built into RSMTool

### Linear models which use the originally defined feature set

* `LinearRegression` -  empirical regression weights using ordinary least squares regression.

* `EqualWeightsLR` all feature weights are set to 1.0; a naive model

#### Linear models with no negative coefficients 

* `RebalancedLR` -  empirical regression weights are rebalanced by using a small portion of positive weights to replace negative beta values.

### Linear models with automatic feature selection

* `LassoFixedLambdaThenLR` - empirical OLS regression weights with feature selection using Lasso regression with all coefficients set to positive. The hyperparameter `lambda` is set to `sqrt(n*lg(p))`. This approach was chosen to balance the penalties for error vs. penalty for two many coefficients to foce Lasso perform more aggressive feature selection, so it may not necessarily achieve the best possible performance. The feature set selected by LASSO is then used to fit an OLS linear regression. Note that while the original Lasso model is constrained to positive coefficients only, small negative coefficients may appear when the coefficients are re-estimated using OLS regression. 

* `PositiveLassoCVThenLR` - empirical OLS regression weights with feature selection using Lasso regression with all coefficients set to positive. The hyperparameter `lambda` is optimized using crossvalidation for loglikehood. The feature set selected by LASSO is then used to fit an OLS linear regression. Note that this approach will likely produce a model with a large N features and any advantages of running Lasso would be effectively negated by latter adding those features to OLS regression. 

### Linear models with automatic feature selection and no negative coefficients
    
* `NNLS` - empirical OLS regression weights with feature selection using non-negative least squares regression. Note that only the coefficients are constrained to be positive: the intercept can be either positive or negative. 

* `LassoFixedLambdaThenNNLS` - empirical OLS regression weights with feature selection using Lasso regression as above. After the features are fit into OLS regression, the coefficients are checked for sign and all features with negative coefficients are removed from selection. The model is then re-fit using the remaining features. Note that this procedure is only applied once to remove features which have very small contribution to the model. This means that there still may be negative coefficients after second attempt. In this case it is advisable to examine the data to identify the best way to proceed. 

* `LassoFixedLambda` - same as `LassoFixedLambdaThenLR` but the model uses the original Lasso weights. Please note: the coefficients in Lasso model are estimated using an optimization routine which may produce slightly different results on different systems. 

* `PositiveLassoCV` - same as `PositiveLassoCVThenLR` but using the original Lasso weights. Please note: the coefficients in Lasso model are estimated using an optimization routine which may produce slightly different results on different systems. 

## Other models available via SKLL

See [SKLL documentation](http://skll.readthedocs.org/en/latest/run_experiment.html#learners) for more information.

* `AdaBoostRegressor`

* `DecisionTreeRegressor`

* `ElasticNet`

* `GradientBoostingRegressor`

* `KNeighborsRegressor`

* `Lasso`

* `LinearSVR`

* `RandomForestRegressor`

* `Ridge`

* `SGDRegressor`

* `SVR`

