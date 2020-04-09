from importlib import import_module
from inspect import isclass, getmembers

_skll_module = import_module('skll.learner')

BUILTIN_MODELS = ['LinearRegression',
                  'EqualWeightsLR',
                  'ScoreWeightedLR',
                  'RebalancedLR',
                  'NNLR',
                  'NNLRIterative',
                  'LassoFixedLambdaThenNNLR',
                  'LassoFixedLambdaThenLR',
                  'PositiveLassoCVThenLR',
                  'LassoFixedLambda',
                  'PositiveLassoCV']

# compute all the classes from sklearn imported into SKLL that have
# an `_estimator_type` attribute - this should give us a list of
# all the SKLL learners that we can support but we need to exclude
# `GridSearchCV` and `Pipeline` which we know are false positives
VALID_SKLL_MODELS = [name for name, member in getmembers(_skll_module)
                     if (isclass(member) and
                         hasattr(member, '_estimator_type') and
                         name not in ['GridSearchCV', 'Pipeline'])]


def is_skll_model(model_name):
    """
    Check whether the given model is a valid learner name in SKLL.
    Note that the `LinearRegression` model is also available in
    SKLL but we always want to use the built-in model with that name.

    Parameters
    ----------
    model_name : str
        The name of the model to check

    Returns
    -------
    valid: bool
        `True` if the given model name is a valid SKLL learner,
        `False` otherwise
    """
    return hasattr(_skll_module, model_name) and model_name != 'LinearRegression'


def is_built_in_model(model_name):
    """
    Check whether the given model is a valid built-in model.

    Parameters
    ----------
    model_name : str
        The name of the model to check

    Returns
    -------
    valid: bool
        `True` if the given model name is a valid built-in model,
        `False` otherwise
    """
    return model_name in BUILTIN_MODELS
