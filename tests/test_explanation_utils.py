from os import environ
from os.path import join

import numpy as np
from nose.tools import assert_equal
from numpy.testing import assert_array_almost_equal, assert_array_equal
from sklearn.datasets import make_classification
from skll.data import FeatureSet, Reader
from skll.learner import Learner

from rsmtool.rsmexplain import mask, select_examples

# allow test directory to be set via an environment variable
# which is needed for package testing
TEST_DIR = environ.get("TESTDIR", None)
if TEST_DIR:
    rsmtool_test_dir = TEST_DIR
else:
    from rsmtool.test_utils import rsmtool_test_dir


class TestExplainUtils:
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

        train_ids = list(range(1, train_size + 1))
        train_features = [
            dict(zip([f"FEATURE_{i + 1}" for i in range(num_features)], x)) for x in X_train
        ]
        train_labels = list(y_train)

        test_ids = list(range(1, test_size + 1))
        test_features = [
            dict(zip([f"FEATURE_{i + 1}" for i in range(num_features)], x)) for x in X_test
        ]

        cls.train_fs = FeatureSet(
            "train", ids=train_ids, features=train_features, labels=train_labels
        )
        cls.test_fs = FeatureSet("test", ids=test_ids, features=test_features)

        # create a dummy learner
        svc = Learner("SVC")
        _ = svc.train(cls.train_fs, grid_search=False)
        cls.svc = svc

    def test_select_features_empty_range_size(self):
        """Test select_features with no user-specified range or size."""
        expected_output = {
            0: 1,
            1: 2,
            2: 3,
            3: 4,
            4: 5,
            5: 6,
            6: 7,
            7: 8,
            8: 9,
            9: 10,
            10: 11,
            11: 12,
            12: 13,
            13: 14,
            14: 15,
        }
        assert_equal(select_examples(self.test_fs), expected_output)

    def test_select_features_integer_range_size(self):
        """Test select_features with an integer range size."""
        expected_output = {5: 6, 12: 13}
        assert_equal(select_examples(self.test_fs, range_size=2), expected_output)

    def test_select_features_full_range_size(self):
        """Test select_features with an ierable range size."""
        expected_output = {5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11}
        assert_equal(select_examples(self.test_fs, range_size=[5, 10]), expected_output)

    def test_select_features_exceed_range_size(self):
        """Test select_features with an exceed range size."""
        expected_output = {10: 11, 11: 12, 12: 13, 13: 14, 14: 15}
        assert_equal(select_examples(self.test_fs, range_size=[10, 20]), expected_output)

    def test_mask_from_learner_in_memory(self):
        """Test mask with a SKLL Learner created in memory."""
        output_path = join(rsmtool_test_dir, "data", "output", "explain_mask_from_learner.out")
        expected_features = np.loadtxt(output_path)
        expected_ids = {5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11}
        computed_ids, computed_features = mask(self.svc, self.test_fs, feature_range=[5, 10])
        assert_equal(computed_ids, expected_ids)
        assert_array_almost_equal(computed_features, expected_features)

    def test_mask_from_learner_on_disk(self):
        """Test mask with a SKLL Learner saved to disk."""
        model_path = join(rsmtool_test_dir, "data", "files", "explain_svr.model")
        data_path = join(rsmtool_test_dir, "data", "files", "explain_features.csv")
        output_path = join(rsmtool_test_dir, "data", "output", "explain_mask_from_pickle.out")
        expected_features = np.loadtxt(output_path)

        model = Learner.from_file(model_path)
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
        assert_equal(computed_ids, expected_ids)
        assert_array_equal(computed_features, expected_features)
