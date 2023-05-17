import json
import pickle
from os import environ
from os.path import join

from nose.tools import assert_equal
from sklearn.datasets import make_classification
from skll.data import FeatureSet, Reader
from skll.learner import Learner

from rsmtool.rsmexplain import mask, yield_ids

# allow test directory to be set via an environment variable
# which is needed for package testing
TEST_DIR = environ.get("TESTDIR", None)
if TEST_DIR:
    rsmtool_test_dir = TEST_DIR
else:
    from rsmtool.test_utils import rsmtool_test_dir


class TestYieldIds:
    @classmethod
    def setUpClass(cls):
        # create a dummy train and test feature set
        X, y = make_classification(
            n_samples=510, n_features=10, n_classes=5, n_informative=8, random_state=123
        )
        X_train, y_train = X[:500], y[:500]
        X_test = X[500:]

        train_ids = list(range(1, len(X_train) + 1))
        train_features = [
            dict(zip([f"FEATURE_{i + 1}" for i in range(X_train.shape[1])], x)) for x in X_train
        ]
        train_labels = list(y_train)

        test_ids = list(range(1, len(X_test) + 1))
        test_features = [
            dict(zip([f"FEATURE_{i + 1}" for i in range(X_test.shape[1])], x)) for x in X_test
        ]

        cls.train_fs = FeatureSet(
            "train", ids=train_ids, features=train_features, labels=train_labels
        )
        cls.test_fs = FeatureSet("test", ids=test_ids, features=test_features)

        # create a dummy learner
        cls.svc = Learner("SVC")
        _ = cls.svc.train(cls.train_fs, grid_search=False)

    def test_yield_ids_empty_range_size(self):
        """Test yield_ids with no range size."""
        expected_output = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10}
        assert_equal(yield_ids(self.test_fs), expected_output)

    def test_yield_ids_integer_range_size(self):
        """Test yield_ids with an integer range size."""
        range_size = 2
        expected_output = {5: 6, 0: 1}
        assert_equal(yield_ids(self.test_fs, range_size), expected_output)

    def test_yield_ids_full_range_size(self):
        """Test yield_ids with an ierable range size."""
        range_size = [5, 10]
        expected_output = {5: 6, 6: 7, 7: 8, 8: 9, 9: 10}
        assert_equal(yield_ids(self.test_fs, range_size), expected_output)

    def test_mask_from_learner(self):
        """Test mask with a SKLL Learner"""
        output_path = join(rsmtool_test_dir, "data", "output", "explain_mask_from_learner.json")
        feature_range = [5, 10]
        expected_ids = {0: 6, 1: 7, 2: 8, 3: 9, 4: 10}
        test_ids, test_features = mask(self.svc, self.test_fs, feature_range)
        assert_equal(test_ids, expected_ids)

        with open(output_path, "r") as fh:
            expected_features = json.load(fh)

        assert_equal(test_features.tolist(), expected_features)


def test_mask_from_pickle():
    """Test mask with a saved model"""
    model_path = join(rsmtool_test_dir, "data", "files", "explain_svc.model")
    data_path = join(rsmtool_test_dir, "data", "files", "explain_features.csv")
    output_path = join(rsmtool_test_dir, "data", "output", "explain_mask_from_pickle.json")

    model = pickle.load(open(model_path, "rb"))
    reader = Reader.for_path(data_path, sparse=False, id_col="id")
    background = reader.read()

    feature_range = [5, 10]
    expected_ids = {0: "EXAMPLE_5", 1: "EXAMPLE_6", 2: "EXAMPLE_7", 3: "EXAMPLE_8", 4: "EXAMPLE_9"}
    test_ids, test_features = mask(model, background, feature_range)
    assert_equal(test_ids, expected_ids)

    with open(output_path, "r") as fh:
        expected_features = json.load(fh)

    assert_equal(test_features.tolist(), expected_features)
