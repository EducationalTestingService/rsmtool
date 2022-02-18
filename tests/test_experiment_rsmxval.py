import os

from parameterized import param, parameterized
from rsmtool.test_utils import check_run_cross_validation

# allow test directory to be set via an environment variable
# which is needed for package testing
TEST_DIR = os.environ.get('TESTDIR', None)
if TEST_DIR:
    rsmtool_test_dir = TEST_DIR
else:
    from rsmtool.test_utils import rsmtool_test_dir


@parameterized([
    param('lr-xval', 'lr_xval'),  # uses 5 folds if not specified
    param('lr-xval-tsv', 'lr_xval_tsv', num_folds=3, file_format="tsv"),
    param('lr-xval-xlsx', 'lr_xval_xlsx', num_folds=3, file_format="xlsx"),
    param('lr-xval-folds-file', 'lr_xval_folds_file', num_folds=2),  # folds file contain 2 folds
    param('lr-xval-subgroups', 'lr_xval_subgroups', num_folds=3, subgroups=["QUESTION", "L1"]),
    param('lr-xval-consistency', 'lr_xval_consistency', num_folds=3, consistency=True, subgroups=["L1"]),
    param('lr-xval-skll-model', 'lr_xval_skll_model', num_folds=2, skll=True),  # uses folds file
    param('lr-xval-thumbnails', 'lr_xval_thumbnails', num_folds=3)
])
def test_run_cross_validation_parameterized(*args, **kwargs):
    if TEST_DIR:
        kwargs['given_test_dir'] = TEST_DIR
    check_run_cross_validation(*args, **kwargs)
