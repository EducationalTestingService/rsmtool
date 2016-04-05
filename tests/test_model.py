from nose.tools import eq_, raises

from rsmtool.model import check_model_name

def test_model_name_builtin_model():
    model_name = 'LinearRegression'
    model_type = check_model_name(model_name)
    eq_(model_type, 'BUILTIN')

def test_model_name_skll_model():
    model_name = 'AdaBoostRegressor'
    model_type = check_model_name(model_name)
    eq_(model_type, 'SKLL')

@raises(ValueError)
def test_model_name_wrong_name():
    model_name = 'random_model'
    _ = check_model_name(model_name)
