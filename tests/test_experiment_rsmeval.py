import os
import tempfile
from os import getcwd
from os.path import join

from nose.tools import raises
from parameterized import param, parameterized

from rsmtool import run_evaluation
from rsmtool.configuration_parser import Configuration
from rsmtool.test_utils import check_run_evaluation, copy_data_files, do_run_evaluation

# allow test directory to be set via an environment variable
# which is needed for package testing
TEST_DIR = os.environ.get("TESTDIR", None)
if TEST_DIR:
    rsmtool_test_dir = TEST_DIR
else:
    from rsmtool.test_utils import rsmtool_test_dir


@parameterized(
    [
        param("lr-eval", "lr_evaluation"),
        param("lr-eval-with-scaling", "lr_evaluation_with_scaling"),
        param(
            "lr-eval-exclude-listwise",
            "lr_eval_exclude_listwise",
            subgroups=["QUESTION", "L1"],
        ),
        param("lr-eval-exclude-flags", "lr_eval_exclude_flags"),
        param(
            "lr-eval-with-missing-scores",
            "lr_eval_with_missing_scores",
            subgroups=["QUESTION", "L1"],
        ),
        param(
            "lr-eval-with-missing-data",
            "lr_eval_with_missing_data",
            subgroups=["QUESTION", "L1"],
        ),
        param("lr-eval-with-custom-order", "lr_eval_with_custom_order", consistency=True),
        param("lr-eval-with-custom-sections", "lr_eval_with_custom_sections"),
        param(
            "lr-eval-with-custom-sections-and-order",
            "lr_eval_with_custom_sections_and_order",
            subgroups=["QUESTION", "L1"],
        ),
        param("lr-eval-tsv-input-files", "lr_eval_tsv_input_files"),
        param("lr-eval-xlsx-input-files", "lr_eval_xlsx_input_files"),
        param("lr-eval-jsonlines-input-files", "lr_eval_jsonlines_input_files"),
        param(
            "lr-eval-nested-jsonlines-input-files",
            "lr_eval_nested_jsonlines_input_files",
        ),
        param("lr-eval-with-tsv-output", "lr_eval_with_tsv_output", file_format="tsv"),
        param("lr-eval-with-xlsx-output", "lr_eval_with_xlsx_output", file_format="xlsx"),
        param(
            "lr-eval-with-h2",
            "lr_eval_with_h2",
            subgroups=["QUESTION", "L1"],
            consistency=True,
        ),
        param("lr-eval-with-h2-named-sc1", "lr_eval_with_h2_named_sc1", consistency=True),
        param(
            "lr-eval-with-scaling-and-h2-keep-zeros",
            "lr_eval_with_scaling_and_h2_keep_zeros",
            consistency=True,
        ),
        param(
            "lr-eval-with-continuous-human-scores",
            "lr_eval_with_continuous_human_scores",
            consistency=True,
        ),
        param(
            "lr-eval-with-subset-double-scored",
            "lr_eval_with_subset_double_scored",
            consistency=True,
        ),
        param("lr-eval-with-trim-tolerance", "lr_evaluation_with_trim_tolerance"),
        param(
            "lr-eval-with-numeric-threshold",
            "lr_evaluation_with_numeric_threshold",
            subgroups=["QUESTION"],
        ),
        param(
            "lr-eval-system-score-constant",
            "lr_eval_system_score_constant",
            subgroups=["QUESTION", "L1"],
            consistency=True,
            suppress_warnings_for=[UserWarning],
        ),
    ]
)
def test_run_experiment_parameterized(*args, **kwargs):
    if TEST_DIR:
        kwargs["given_test_dir"] = TEST_DIR
    check_run_evaluation(*args, **kwargs)


def test_run_experiment_lr_eval_with_object():
    """Test rsmeval using a Configuration object, rather than a file."""
    source = "lr-eval-object"
    experiment_id = "lr_eval_object"

    configdir = join(rsmtool_test_dir, "data", "experiments", source)

    config_dict = {
        "predictions_file": "../../files/predictions_scaled_with_subgroups.csv",
        "system_score_column": "score",
        "description": "An evaluation of LinearRegression predictions.",
        "human_score_column": "h1",
        "id_column": "id",
        "experiment_id": "lr_eval_object",
        "subgroups": "QUESTION",
        "scale_with": "asis",
        "trim_min": 1,
        "trim_max": 6,
    }

    config_obj = Configuration(config_dict, context="rsmeval", configdir=configdir)

    check_run_evaluation(source, experiment_id, config_obj_or_dict=config_obj)


def test_run_experiment_lr_eval_with_dictionary():
    """Test rsmeval using the dictionary object, rather than a file."""
    source = "lr-eval-dict"
    experiment_id = "lr_eval_dict"

    # set up a temporary directory since
    # we will be using getcwd
    temp_dir = tempfile.TemporaryDirectory(prefix=getcwd())

    old_file_dict = {"pred": "data/files/predictions_scaled_with_subgroups.csv"}

    new_file_dict = copy_data_files(temp_dir.name, old_file_dict, rsmtool_test_dir)

    config_dict = {
        "predictions_file": new_file_dict["pred"],
        "system_score_column": "score",
        "description": "An evaluation of LinearRegression predictions.",
        "human_score_column": "h1",
        "id_column": "id",
        "experiment_id": "lr_eval_dict",
        "subgroups": "QUESTION",
        "scale_with": "asis",
        "trim_min": 1,
        "trim_max": 6,
    }

    check_run_evaluation(source, experiment_id, config_obj_or_dict=config_dict)


@raises(ValueError)
def test_run_evaluation_wrong_input_format():
    config_list = [("experiment_id", "AAAA"), ("train_file", "some_path")]
    with tempfile.TemporaryDirectory() as temp_dir:
        run_evaluation(config_list, temp_dir)


@raises(ValueError)
def test_run_experiment_lr_eval_with_repeated_ids():

    # rsmeval experiment with non-unique ids
    source = "lr-eval-with-repeated-ids"
    experiment_id = "lr_eval_with_repeated_ids"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")
    do_run_evaluation(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_eval_all_non_numeric_scores():

    # rsmeval experiment with all values for the human
    # score being non-numeric and all getting filtered out
    # which should raise an exception

    source = "lr-eval-with-all-non-numeric-scores"
    experiment_id = "lr_eval_all_non_numeric_scores"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")
    do_run_evaluation(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_eval_same_system_human_score():

    # rsmeval experiment with the same value supplied
    # for both human score ans system score

    source = "lr-eval-same-system-human-score"
    experiment_id = "lr_eval_same_system_human_score"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")
    do_run_evaluation(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_eval_all_non_numeric_machine_scores():

    # rsmeval experiment with all the machine scores`
    # being non-numeric and all getting filtered out
    # which should raise an exception

    source = "lr-eval-with-all-non-numeric-machine-scores"
    experiment_id = "lr_eval_all_non_numeric_machine_scores"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")
    do_run_evaluation(source, experiment_id, config_file)


@raises(KeyError)
def test_run_experiment_eval_lr_with_missing_h2_column():

    # rsmeval experiment with `second_human_score_column`
    # set to a column that does not exist in the given
    # predictions file
    source = "lr-eval-with-missing-h2-column"
    experiment_id = "lr_eval_with_missing_h2_column"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")
    do_run_evaluation(source, experiment_id, config_file)


@raises(KeyError)
def test_run_experiment_eval_lr_with_missing_candidate_column():

    # rsmeval experiment with `candidate_column`
    # set to a column that does not exist in the given
    # predictions file
    source = "lr-eval-with-missing-candidate-column"
    experiment_id = "lr_eval_with_missing_candidate_column"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")
    do_run_evaluation(source, experiment_id, config_file)


@raises(FileNotFoundError)
def test_run_experiment_lr_eval_wrong_path():

    # basic rsmeval experiment with wrong path to the
    # predictions file

    source = "lr-eval-with-wrong-path"
    experiment_id = "lr_eval_with_h2"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")
    do_run_evaluation(source, experiment_id, config_file)
