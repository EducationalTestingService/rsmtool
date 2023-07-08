import os
import unittest

from nose2.tools import params

from rsmtool.test_utils import check_run_cross_validation

# allow test directory to be set via an environment variable
# which is needed for package testing
TEST_DIR = os.environ.get("TESTDIR", None)
if TEST_DIR:
    rsmtool_test_dir = TEST_DIR
else:
    from rsmtool.test_utils import rsmtool_test_dir  # noqa


class TestExperimentRsmxval(unittest.TestCase):
    @params(
        {"source": "lr-xval", "experiment_id": "lr_xval"},
        {"source": "lr-xval-tsv", "experiment_id": "lr_xval_tsv", "folds": 3, "file_format": "tsv"},
        {
            "source": "lr-xval-xlsx",
            "experiment_id": "lr_xval_xlsx",
            "folds": 3,
            "file_format": "xlsx",
        },
        {"source": "lr-xval-folds-file", "experiment_id": "lr_xval_folds_file", "folds": 2},
        {
            "source": "lr-xval-subgroups",
            "experiment_id": "lr_xval_subgroups",
            "folds": 3,
            "subgroups": ["QUESTION", "L1"],
        },
        {
            "source": "lr-xval-subgroups-with-int-ids",
            "experiment_id": "lr_xval_subgroups_with_int_ids",
            "folds": 3,
            "subgroups": ["QUESTION", "L1"],
        },
        {
            "source": "lr-xval-consistency",
            "experiment_id": "lr_xval_consistency",
            "folds": 3,
            "consistency": True,
            "subgroups": ["L1"],
        },
        {
            "source": "lr-xval-skll-model",
            "experiment_id": "lr_xval_skll_model",
            "folds": 2,
            "skll": True,
        },
        {"source": "lr-xval-thumbnails", "experiment_id": "lr_xval_thumbnails", "folds": 3},
        {"source": "lr-xval-feature-list", "experiment_id": "lr_xval_feature_list", "folds": 3},
        {
            "source": "lr-xval-feature-subset-file",
            "experiment_id": "lr_xval_feature_subset_file",
            "folds": 3,
        },
    )
    def test_run_cross_validation_parameterized(self, kwargs):
        if TEST_DIR:
            kwargs["given_test_dir"] = TEST_DIR
        check_run_cross_validation(**kwargs)
