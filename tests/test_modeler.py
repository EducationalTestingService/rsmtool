import numpy as np
import pandas as pd
from nose.tools import eq_
from numpy.testing import assert_array_equal, assert_raises_regex

from rsmtool.modeler import Modeler


class TestModeler:
    def setUp(self):

        series = pd.Series([34, 0.34, 1.2], index=["const", "A", "B"])
        coef = Modeler().ols_coefficients_to_dataframe(series)
        learner = Modeler().create_fake_skll_learner(coef)
        self.modeler = Modeler.load_from_learner(learner)

    def test_get_coefficients(self):
        coefficients = self.modeler.get_coefficients()
        assert_array_equal(coefficients, np.array([0.34, 1.2]))

    def test_get_coefficients_is_none(self):
        modeler = Modeler()
        eq_(modeler.get_coefficients(), None)

    def test_get_intercept(self):
        intercept = self.modeler.get_intercept()
        eq_(intercept, 34)

    def test_get_intercept_is_none(self):
        modeler = Modeler()
        eq_(modeler.get_intercept(), None)

    def test_get_feature_names(self):
        intercept = self.modeler.get_feature_names()
        eq_(intercept, ["A", "B"])

    def test_get_feature_names_is_none(self):
        modeler = Modeler()
        eq_(modeler.get_feature_names(), None)

    def test_expected_scores_no_min_max(self):
        df = pd.DataFrame([{"spkitemid": "DUMMY", "A": 1, "B": 2}])
        with assert_raises_regex(
            ValueError, r"Must specify 'min_score' and 'max_score' for expected scores."
        ):
            self.modeler.predict(df, predict_expected=True)
