import os

from parameterized import param, parameterized

from rsmtool.test_utils import check_run_explain

# allow test directory to be set via an environment variable
# which is needed for package testing
TEST_DIR = os.environ.get("TESTDIR", None)
if TEST_DIR:
    rsmtool_test_dir = TEST_DIR
else:
    from rsmtool.test_utils import rsmtool_test_dir  # noqa: F401


@parameterized(
    [
        param("svc-explain", "svc_explain"),
        param("knn-explain", "knn_explain"),
        param("bay-explain", "bay_explain"),
    ]
)
def test_run_experiment_parameterized(*args, **kwargs):
    if TEST_DIR:
        kwargs["given_test_dir"] = TEST_DIR
    check_run_explain(*args, **kwargs)
