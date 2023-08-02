import os
import tempfile
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal, assert_raises_regex

from rsmtool.modeler import Modeler


class TestModeler(unittest.TestCase):
    """Tests for Modeler"""

    @classmethod
    def setUpClass(self):
        series = pd.Series([34, 0.34, 1.2], index=["const", "A", "B"])
        coef = Modeler().ols_coefficients_to_dataframe(series)
        learner = Modeler().create_fake_skll_learner(coef)
        self.modeler = Modeler.load_from_learner(learner)

    def test_get_coefficients(self):
        coefficients = self.modeler.get_coefficients()
        assert_array_equal(coefficients, np.array([0.34, 1.2]))

    def test_get_coefficients_is_none(self):
        modeler = Modeler()
        self.assertEqual(modeler.get_coefficients(), None)

    def test_get_intercept(self):
        intercept = self.modeler.get_intercept()
        self.assertEqual(intercept, 34)

    def test_get_intercept_is_none(self):
        modeler = Modeler()
        self.assertEqual(modeler.get_intercept(), None)

    def test_get_feature_names(self):
        intercept = self.modeler.get_feature_names()
        self.assertEqual(intercept, ["A", "B"])

    def test_get_feature_names_is_none(self):
        modeler = Modeler()
        self.assertEqual(modeler.get_feature_names(), None)

    def test_expected_scores_no_min_max(self):
        df = pd.DataFrame([{"spkitemid": "DUMMY", "A": 1, "B": 2}])
        with assert_raises_regex(
            ValueError, r"Must specify 'min_score' and 'max_score' for expected scores."
        ):
            self.modeler.predict(df, predict_expected=True)

    def test_save_and_load(self):
        """Test that a modeler can be saved and subsequently loaded."""
        model_file = tempfile.NamedTemporaryFile(suffix=".model", delete=False)
        model_file.close()
        self.modeler.save(model_file.name)
        Modeler.load_from_file(model_file.name)
        os.unlink(model_file.name)

    def test_load_from_skll_model_file(self):
        """
        Test that SKLL learners can be loaded as modelers.

        Meant to test for backward compatibility. Saved models were
        formerly bare SKLL learner objects. This can be emulated by
        saving just the learner attribute within the modeler object
        and then trying to load that in the same way modeler objects
        would be loaded.
        """
        model_file = tempfile.NamedTemporaryFile(suffix=".model", delete=False)
        model_file.close()
        self.modeler.learner.save(model_file.name)
        Modeler.load_from_file(model_file.name)
        os.unlink(model_file.name)
