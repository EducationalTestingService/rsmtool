import os
from os.path import join

from nose.tools import raises
from parameterized import param, parameterized

from rsmtool.test_utils import check_run_experiment, do_run_experiment

# allow test directory to be set via an environment variable
# which is needed for package testing
TEST_DIR = os.environ.get("TESTDIR", None)
if TEST_DIR:
    rsmtool_test_dir = TEST_DIR
else:
    from rsmtool.test_utils import rsmtool_test_dir


@parameterized(
    [
        param(
            "lr-with-defaults-as-extra-columns",
            "lr_with_defaults_as_extra_columns",
            consistency=True,
        ),
        param("lr-with-truncations", "lr_with_truncations"),
        param("lr-exclude-listwise", "lr_exclude_listwise"),
        param("lr-with-custom-order", "lr_with_custom_order"),
        param("lr-with-custom-sections", "lr_with_custom_sections"),
        param("lr-with-custom-sections-and-order", "lr_with_custom_sections_and_order"),
        param("lr-exclude-flags", "lr_exclude_flags"),
        param("lr-exclude-flags-and-zeros", "lr_exclude_flags_and_zeros"),
        param("lr-use-all-features", "lr_use_all_features"),
        param("lr-candidate-same-as-id", "lr_candidate_same_as_id"),
        param("lr-candidate-same-as-id-candidate", "lr_candidate_same_as_id_candidate"),
        param("lr-tsv-input-files", "lr_tsv_input_files"),
        param("lr-tsv-input-and-subset-files", "lr_tsv_input_and_subset_files"),
        param("lr-xlsx-input-files", "lr_xlsx_input_files"),
        param("lr-xlsx-input-and-subset-files", "lr_xlsx_input_and_subset_files"),
        param(
            "lr-with-subset-double-scored",
            "lr_with_subset_double_scored",
            consistency=True,
        ),
        param("lr-jsonlines-input-files", "lr_jsonlines_input_files"),
        param("lr-nested-jsonlines-input-files", "lr_nested_jsonlines_input_files"),
    ]
)
def test_run_experiment_parameterized(*args, **kwargs):
    if TEST_DIR:
        kwargs["given_test_dir"] = TEST_DIR
    check_run_experiment(*args, **kwargs)


@raises(ValueError)
def test_run_experiment_lr_length_column_and_feature():

    # rsmtool experiment that has length column but
    # the same column as a model feature
    source = "lr-with-length-and-feature"
    experiment_id = "lr_with_length_and_feature"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")
    do_run_experiment(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_h2_column_and_feature():

    # rsmtool experiment that has second rater column but
    # the same column as a model feature
    source = "lr-with-h2-and-feature"
    experiment_id = "lr_with_h2_and_feature"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")
    do_run_experiment(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_same_h1_and_h2():

    # rsmtool experiment that has label column
    # and second rater column set the same

    source = "lr-same-h1-and-h2"
    experiment_id = "lr_same_h1_and_h2"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")
    do_run_experiment(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_with_repeated_ids():

    # rsmtool experiment with non-unique ids
    source = "lr-with-repeated-ids"
    experiment_id = "lr_with_repeated_ids"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")
    do_run_experiment(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_with_sc1_as_feature_name():

    # rsmtool experiment with sc1 used as the name of a feature
    source = "lr-with-sc1-as-feature-name"
    experiment_id = "lr_with_sc1_as_feature_name"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")
    do_run_experiment(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_with_length_as_feature_name():

    # rsmtool experiment with 'length' used as feature name
    # when a length analysis is requested using a different feature
    source = "lr-with-length-as-feature-name"
    experiment_id = "lr_with_length_as_feature_name"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")
    do_run_experiment(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_with_truncations_no_features_field():

    # rsmtool experiment with truncations, but no feature field
    source = "lr-with-truncations-no-features-field"
    experiment_id = "lr_with_truncations_no_features_field"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")
    do_run_experiment(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_with_truncations_no_features_columns():

    # rsmtool experiment with truncations, but no min/max columns in feature file
    source = "lr-with-truncations-no-features-columns"
    experiment_id = "lr_with_truncations_no_features_columns"
    config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")
    do_run_experiment(source, experiment_id, config_file)
