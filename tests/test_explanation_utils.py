import unittest
from os import environ
from os.path import join

import numpy as np
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.datasets import make_classification
from skll.data import FeatureSet, Reader
from skll.learner import Learner

from rsmtool.modeler import Modeler
from rsmtool.rsmexplain import mask, select_examples

# allow test directory to be set via an environment variable
# which is needed for package testing
TEST_DIR = environ.get("TESTDIR", None)
if TEST_DIR:
    rsmtool_test_dir = TEST_DIR
else:
    from rsmtool.test_utils import rsmtool_test_dir


class TestExplainUtils(unittest.TestCase):
    """Test class for Explain Utils tests."""

    @classmethod
    def setUpClass(cls):
        # create a dummy train and test feature set
        total_size = 515
        train_size = 500
        test_size = total_size - train_size
        num_features = 10
        X, y = make_classification(
            n_samples=total_size,
            n_features=num_features,
            n_classes=5,
            n_informative=8,
            random_state=123,
        )
        X_train, y_train = X[:train_size], y[:train_size]
        X_test = X[train_size:]

        train_ids = [f"TRAIN_{idx}" for idx in range(1, train_size + 1)]
        train_features = [
            dict(zip([f"FEATURE_{i + 1}" for i in range(num_features)], x)) for x in X_train
        ]
        train_labels = list(y_train)

        test_ids_strings = [f"TEST_{idx}" for idx in range(1, test_size + 1)]
        test_ids_ints = list(range(1, test_size + 1))
        test_features = [
            dict(zip([f"FEATURE_{i + 1}" for i in range(num_features)], x)) for x in X_test
        ]

        cls.train_fs = FeatureSet(
            "train", ids=train_ids, features=train_features, labels=train_labels
        )
        cls.test_fs = FeatureSet("test", ids=test_ids_strings, features=test_features)
        cls.test_fs_int_ids = FeatureSet("test", ids=test_ids_ints, features=test_features)

        # create a dummy learner
        svc = Learner("SVC")
        _ = svc.train(cls.train_fs, grid_search=False)
        cls.svc = svc

    def test_select_features_empty_range_size(self):
        """Test select_features with no user-specified range or size."""
        expected_output = {
            0: "TEST_1",
            1: "TEST_2",
            2: "TEST_3",
            3: "TEST_4",
            4: "TEST_5",
            5: "TEST_6",
            6: "TEST_7",
            7: "TEST_8",
            8: "TEST_9",
            9: "TEST_10",
            10: "TEST_11",
            11: "TEST_12",
            12: "TEST_13",
            13: "TEST_14",
            14: "TEST_15",
        }
        self.assertEqual(select_examples(self.test_fs), expected_output)

    def test_select_features_integer_range_size(self):
        """Test select_features with an integer range size."""
        expected_output = {1: "TEST_2", 6: "TEST_7"}
        self.assertEqual(select_examples(self.test_fs, range_size=2), expected_output)

    def test_select_features_integer_exceed_range_size(self):
        """Test select_features with range size larger than data size."""
        expected_output = {
            0: "TEST_1",
            1: "TEST_2",
            2: "TEST_3",
            3: "TEST_4",
            4: "TEST_5",
            5: "TEST_6",
            6: "TEST_7",
            7: "TEST_8",
            8: "TEST_9",
            9: "TEST_10",
            10: "TEST_11",
            11: "TEST_12",
            12: "TEST_13",
            13: "TEST_14",
            14: "TEST_15",
        }
        self.assertEqual(select_examples(self.test_fs, range_size=20), expected_output)

    def test_select_features_full_range_size(self):
        """Test select_features with an iterable range size."""
        expected_output = {
            5: "TEST_6",
            6: "TEST_7",
            7: "TEST_8",
            8: "TEST_9",
            9: "TEST_10",
            10: "TEST_11",
        }
        self.assertEqual(select_examples(self.test_fs, range_size=[5, 10]), expected_output)

    def test_select_features_exceed_range_size(self):
        """Test select_features with range size larger than data size."""
        expected_output = {
            10: "TEST_11",
            11: "TEST_12",
            12: "TEST_13",
            13: "TEST_14",
            14: "TEST_15",
        }
        self.assertEqual(select_examples(self.test_fs, range_size=[10, 20]), expected_output)

    def test_select_features_range_ids_size(self):
        """Test select_features with specific example IDs."""
        expected_output = {4: "TEST_5", 9: "TEST_10", 11: "TEST_12"}
        self.assertEqual(
            select_examples(self.test_fs, range_size=("TEST_5", "TEST_10", "TEST_12")),
            expected_output,
        )

    def test_select_features_inordered_range_ids_size(self):
        """Test select_features with unordered range ids."""
        expected_output = {11: "TEST_12", 4: "TEST_5", 9: "TEST_10"}
        self.assertEqual(
            select_examples(self.test_fs, range_size=("TEST_12", "TEST_5", "TEST_10")),
            expected_output,
        )

    def test_select_features_mismatched_range_ids_size(self):
        """Test select_features with range IDs as strings but featureset IDs as ints."""
        expected_output = {4: 5, 9: 10, 11: 12}
        self.assertEqual(
            select_examples(self.test_fs_int_ids, range_size=("5", "10", "12")),
            expected_output,
        )

    def test_select_features_exceed_range_ids_size(self):
        """Test select_features with out of boundary range ids."""
        with self.assertRaises(ValueError):
            select_examples(self.test_fs, range_size=("TEST_5", "TEST_10", "TEST_20"))

    def test_select_features_invalid_range_ids_size(self):
        """Test select_features with invalid range ids."""
        with self.assertRaises(ValueError):
            select_examples(self.test_fs, range_size=("1", "5", "10"))

    def test_mask_from_learner_in_memory(self):
        """Test mask with a SKLL Learner created in memory."""
        output_path = join(rsmtool_test_dir, "data", "output", "explain_mask_from_learner.out")
        expected_features = np.loadtxt(output_path)
        expected_ids = {
            5: "TEST_6",
            6: "TEST_7",
            7: "TEST_8",
            8: "TEST_9",
            9: "TEST_10",
            10: "TEST_11",
        }
        computed_ids, computed_features = mask(self.svc, self.test_fs, feature_range=[5, 10])
        self.assertEqual(computed_ids, expected_ids)
        assert_array_almost_equal(computed_features, expected_features)

    def test_mask_from_learner_on_disk(self):
        """Test mask with a SKLL Learner saved to disk."""
        model_path = join(rsmtool_test_dir, "data", "files", "explain_svr.model")
        data_path = join(rsmtool_test_dir, "data", "files", "explain_features.csv")
        output_path = join(rsmtool_test_dir, "data", "output", "explain_mask_from_pickle.out")
        expected_features = np.loadtxt(output_path)

        model = Modeler.load_from_file(model_path).learner
        reader = Reader.for_path(data_path, id_col="ID")
        background = reader.read()

        expected_ids = {
            5: "RESPONSE_6",
            6: "RESPONSE_7",
            7: "RESPONSE_8",
            8: "RESPONSE_9",
            9: "RESPONSE_10",
            10: "RESPONSE_11",
        }
        computed_ids, computed_features = mask(model, background, feature_range=[5, 10])
        self.assertEqual(computed_ids, expected_ids)
        assert_array_equal(computed_features, expected_features)
