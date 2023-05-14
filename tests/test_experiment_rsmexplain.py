import os
import tempfile
from os import getcwd
from os.path import join

from parameterized import param, parameterized

from rsmtool.configuration_parser import Configuration
from rsmtool.test_utils import check_run_explain, copy_data_files

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


def test_run_experiment_svc_explain_with_object():
    """Test rsmexplain using a Configuration object, rather than a file."""
    source = "svc-explain-object"
    experiment_id = "svc_explain_object"

    configdir = join(rsmtool_test_dir, "data", "experiments", source)

    config_dict = {
        "model_path": "../../files/explain_svc.model",
        "background_data": "../../files/explain_features.csv",
        "background_size": "50",
        "explainable_data": "../../files/explain_features.csv",
        "id_column": "id",
        "range": 10,
        "experiment_id": "svc_explain_object",
        "description": "This is a SVR scoring model for illustration purposes",
        "display_num": "16",
        "auto_cohorts": "True",
    }

    config_obj = Configuration(config_dict, context="rsmexplain", configdir=configdir)

    check_run_explain(source, experiment_id, config_obj_or_dict=config_obj)


def test_run_experiment_svc_explain_with_dictionary():
    """Test rsmexplain using the dictionary object, rather than a file."""
    source = "svc-explain-dict"
    experiment_id = "svc_explain_dict"

    # set up a temporary directory since
    # we will be using getcwd
    temp_dir = tempfile.TemporaryDirectory(prefix=getcwd())

    old_file_dict = {
        "model": "data/files/explain_svc.model",
        "background_data": "data/files/explain_features.csv",
    }

    new_file_dict = copy_data_files(temp_dir.name, old_file_dict, rsmtool_test_dir)

    config_dict = {
        "model_path": new_file_dict["model"],
        "background_data": new_file_dict["background_data"],
        "background_size": "50",
        "explainable_data": new_file_dict["background_data"],
        "id_column": "id",
        "range": 10,
        "experiment_id": "svc_explain_dict",
        "description": "This is a SVR scoring model for illustration purposes",
        "display_num": "16",
        "auto_cohorts": "True",
    }

    check_run_explain(source, experiment_id, config_obj_or_dict=config_dict)
