import os
import unittest
from os.path import join

from nose2.tools import params

from rsmtool.test_utils import check_run_experiment, do_run_experiment

# allow test directory to be set via an environment variable
# which is needed for package testing
TEST_DIR = os.environ.get("TESTDIR", None)
if TEST_DIR:
    rsmtool_test_dir = TEST_DIR
else:
    from rsmtool.test_utils import rsmtool_test_dir


class TestExperimentRsmtool2(unittest.TestCase):
    @params(
        {
            "source": "lr-with-defaults-as-extra-columns",
            "experiment_id": "lr_with_defaults_as_extra_columns",
            "consistency": "True",
        },
        {"source": "lr-with-truncations", "experiment_id": "lr_with_truncations"},
        {"source": "lr-exclude-listwise", "experiment_id": "lr_exclude_listwise"},
        {"source": "lr-with-custom-order", "experiment_id": "lr_with_custom_order"},
        {"source": "lr-with-custom-sections", "experiment_id": "lr_with_custom_sections"},
        {
            "source": "lr-with-custom-sections-and-order",
            "experiment_id": "lr_with_custom_sections_and_order",
        },
        {"source": "lr-exclude-flags", "experiment_id": "lr_exclude_flags"},
        {"source": "lr-exclude-flags-and-zeros", "experiment_id": "lr_exclude_flags_and_zeros"},
        {"source": "lr-use-all-features", "experiment_id": "lr_use_all_features"},
        {"source": "lr-candidate-same-as-id", "experiment_id": "lr_candidate_same_as_id"},
        {
            "source": "lr-candidate-same-as-id-candidate",
            "experiment_id": "lr_candidate_same_as_id_candidate",
        },
        {"source": "lr-tsv-input-files", "experiment_id": "lr_tsv_input_files"},
        {
            "source": "lr-tsv-input-and-subset-files",
            "experiment_id": "lr_tsv_input_and_subset_files",
        },
        {"source": "lr-xlsx-input-files", "experiment_id": "lr_xlsx_input_files"},
        {
            "source": "lr-xlsx-input-and-subset-files",
            "experiment_id": "lr_xlsx_input_and_subset_files",
        },
        {
            "source": "lr-with-subset-double-scored",
            "experiment_id": "lr_with_subset_double_scored",
            "consistency": "True",
        },
        {"source": "lr-jsonlines-input-files", "experiment_id": "lr_jsonlines_input_files"},
        {
            "source": "lr-nested-jsonlines-input-files",
            "experiment_id": "lr_nested_jsonlines_input_files",
        },
    )
    def test_run_experiment_parameterized(self, kwargs):
        if TEST_DIR:
            kwargs["given_test_dir"] = TEST_DIR
        check_run_experiment(**kwargs)

    def test_run_experiment_lr_length_column_and_feature(self):
        # rsmtool experiment that has length column but
        # the same column as a model feature
        source = "lr-with-length-and-feature"
        experiment_id = "lr_with_length_and_feature"
        config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")
        with self.assertRaises(ValueError):
            do_run_experiment(source, experiment_id, config_file)

    def test_run_experiment_lr_h2_column_and_feature(self):
        # rsmtool experiment that has second rater column but
        # the same column as a model feature
        source = "lr-with-h2-and-feature"
        experiment_id = "lr_with_h2_and_feature"
        config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")
        with self.assertRaises(ValueError):
            do_run_experiment(source, experiment_id, config_file)

    def test_run_experiment_lr_same_h1_and_h2(self):
        # rsmtool experiment that has label column
        # and second rater column set the same

        source = "lr-same-h1-and-h2"
        experiment_id = "lr_same_h1_and_h2"
        config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")
        with self.assertRaises(ValueError):
            do_run_experiment(source, experiment_id, config_file)

    def test_run_experiment_lr_with_repeated_ids(self):
        # rsmtool experiment with non-unique ids
        source = "lr-with-repeated-ids"
        experiment_id = "lr_with_repeated_ids"
        config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")
        with self.assertRaises(ValueError):
            do_run_experiment(source, experiment_id, config_file)

    def test_run_experiment_lr_with_sc1_as_feature_name(self):
        # rsmtool experiment with sc1 used as the name of a feature
        source = "lr-with-sc1-as-feature-name"
        experiment_id = "lr_with_sc1_as_feature_name"
        config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")
        with self.assertRaises(ValueError):
            do_run_experiment(source, experiment_id, config_file)

    def test_run_experiment_lr_with_length_as_feature_name(self):
        # rsmtool experiment with 'length' used as feature name
        # when a length analysis is requested using a different feature
        source = "lr-with-length-as-feature-name"
        experiment_id = "lr_with_length_as_feature_name"
        config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")
        with self.assertRaises(ValueError):
            do_run_experiment(source, experiment_id, config_file)

    def test_run_experiment_lr_with_truncations_no_features_field(self):
        # rsmtool experiment with truncations, but no feature field
        source = "lr-with-truncations-no-features-field"
        experiment_id = "lr_with_truncations_no_features_field"
        config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")
        with self.assertRaises(ValueError):
            do_run_experiment(source, experiment_id, config_file)

    def test_run_experiment_lr_with_truncations_no_features_columns(self):
        # rsmtool experiment with truncations, but no min/max columns in feature file
        source = "lr-with-truncations-no-features-columns"
        experiment_id = "lr_with_truncations_no_features_columns"
        config_file = join(rsmtool_test_dir, "data", "experiments", source, f"{experiment_id}.json")
        with self.assertRaises(ValueError):
            do_run_experiment(source, experiment_id, config_file)
