import unittest
import warnings

import numpy as np
from numpy.testing import assert_array_equal

from rsmtool.transformer import FeatureTransformer


class TestFeatureTransformer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ft = FeatureTransformer()

    def test_apply_inverse_transform(self):
        self.assertRaises(ValueError, self.ft.apply_inverse_transform, "name", np.array([0, 1, 2]))
        self.assertRaises(
            ValueError,
            self.ft.apply_inverse_transform,
            "name",
            np.array([-2, -3, 1, 2]),
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            assert_array_equal(
                self.ft.apply_inverse_transform("name", np.array([0, 2, 4]), raise_error=False),
                np.array([np.inf, 0.5, 0.25]),
            )
        assert_array_equal(
            self.ft.apply_inverse_transform("name", np.array([-2, -4, 1]), raise_error=False),
            np.array([-0.5, -0.25, 1]),
        )
        assert_array_equal(
            self.ft.apply_inverse_transform("name", np.array([2, 4])),
            np.array([0.5, 0.25]),
        )
        assert_array_equal(
            self.ft.apply_inverse_transform("name", np.array([-2, -4])),
            np.array([-0.5, -0.25]),
        )

    def test_apply_sqrt_transform(self):
        self.assertRaises(
            ValueError, self.ft.apply_sqrt_transform, "name", np.array([-2, -3, 1, 2])
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            assert_array_equal(
                self.ft.apply_sqrt_transform("name", np.array([-1, 2, 4]), raise_error=False),
                np.array([np.nan, np.sqrt(2), 2]),
            )
        assert_array_equal(
            self.ft.apply_sqrt_transform("name", np.array([2, 4])),
            np.array([np.sqrt(2), 2]),
        )
        assert_array_equal(
            self.ft.apply_sqrt_transform("name", np.array([0.5, 4])),
            np.array([np.sqrt(0.5), 2]),
        )
        assert_array_equal(self.ft.apply_sqrt_transform("name", np.array([0, 4])), np.array([0, 2]))

    def test_apply_log_transform(self):
        self.assertRaises(ValueError, self.ft.apply_log_transform, "name", np.array([-1, 2, 3]))
        self.assertRaises(ValueError, self.ft.apply_log_transform, "name", np.array([0, 2, 3]))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            assert_array_equal(
                self.ft.apply_log_transform("name", np.array([-1, 1, 4]), raise_error=False),
                np.array([np.nan, np.log(1), np.log(4)]),
            )
            assert_array_equal(
                self.ft.apply_log_transform("name", np.array([0, 1, 4]), raise_error=False),
                np.array([-np.inf, np.log(1), np.log(4)]),
            )
        assert_array_equal(
            self.ft.apply_log_transform("name", np.array([1, 4])),
            np.array([np.log(1), np.log(4)]),
        )

    def test_apply_add_one_inverse_transform(self):
        self.assertRaises(
            ValueError,
            self.ft.apply_add_one_inverse_transform,
            "name",
            np.array([-1, -2, 3, 5]),
        )
        assert_array_equal(
            self.ft.apply_add_one_inverse_transform(
                "name", np.array([-2, -3, 1, 4]), raise_error=False
            ),
            np.array([-1, -1 / 2, 1 / 2, 1 / 5]),
        )
        assert_array_equal(
            self.ft.apply_add_one_inverse_transform("name", np.array([1, 4])),
            np.array([1 / 2, 1 / 5]),
        )
        assert_array_equal(
            self.ft.apply_add_one_inverse_transform("name", np.array([0, 4])),
            np.array([1, 1 / 5]),
        )

    def test_apply_add_one_log_transform(self):
        self.assertRaises(
            ValueError,
            self.ft.apply_add_one_log_transform,
            "name",
            np.array([-2, -3, 2, 3]),
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            assert_array_equal(
                self.ft.apply_add_one_log_transform(
                    "name", np.array([-2, -0.5, 2, 4]), raise_error=False
                ),
                np.array([np.nan, np.log(0.5), np.log(3), np.log(5)]),
            )
        assert_array_equal(
            self.ft.apply_add_one_log_transform("name", np.array([2, 4])),
            np.array([np.log(3), np.log(5)]),
        )
        assert_array_equal(
            self.ft.apply_add_one_log_transform("name", np.array([0, 4])),
            np.array([0, np.log(5)]),
        )

    def test_transform_feature(self):
        name = "dpsec"
        data = np.array([1, 2, 3, 4])

        # run the test but suppress the expected runtime warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            self.assertRaises(ValueError, self.ft.transform_feature, data, name, "add_one_inverse")
            assert_array_equal(self.ft.transform_feature(data, name, "inv"), 1 / data)
            assert_array_equal(self.ft.transform_feature(data, name, "raw"), data)
            assert_array_equal(self.ft.transform_feature(data, name, "org"), data)
            assert_array_equal(self.ft.transform_feature(data, name, "log"), np.log(data))
            assert_array_equal(self.ft.transform_feature(data, name, "addOneInv"), 1 / (data + 1))
            assert_array_equal(self.ft.transform_feature(data, name, "addOneLn"), np.log(data + 1))

    def test_transform_feature_with_warning(self):
        name = "dpsec"
        data = np.array([-1, 0, 2, 3])

        # run the test but suppress the expected runtime warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            assert_array_equal(
                self.ft.transform_feature(data, name, "sqrt", raise_error=False),
                np.sqrt(data),
            )
            assert_array_equal(
                self.ft.transform_feature(data, name, "inv", raise_error=False),
                1 / data,
            )
            assert_array_equal(
                self.ft.transform_feature(data, name, "addOneInv", raise_error=False),
                1 / (data + 1),
            )
            assert_array_equal(
                self.ft.transform_feature(data, name, "log", raise_error=False),
                np.log(data),
            )
            assert_array_equal(
                self.ft.transform_feature(data, name, "addOneLn", raise_error=False),
                np.log(data + 1),
            )

    def test_transform_feature_with_error(self):
        name = "dpsec"
        data = np.array([-1, 0, 2, 3])

        # run the test but suppress the expected runtime warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            self.assertRaises(ValueError, self.ft.transform_feature, data, name, "sqrt")
            self.assertRaises(ValueError, self.ft.transform_feature, data, name, "inv")
            self.assertRaises(ValueError, self.ft.transform_feature, data, name, "addOneInv")
            self.assertRaises(ValueError, self.ft.transform_feature, data, name, "log")
            self.assertRaises(ValueError, self.ft.transform_feature, data, name, "addOneLn")
