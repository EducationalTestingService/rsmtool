import unittest
import warnings
from os.path import dirname, join

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from scipy.stats import pearsonr

from rsmtool.comparer import Comparer
from rsmtool.test_utils import do_run_experiment


class TestComparer(unittest.TestCase):
    """Test class for Comparer tests"""

    def test_make_summary_stat_df(self):
        array = np.random.multivariate_normal([100, 25], [[25, 0.5], [0.5, 1]], 5000)

        df = pd.DataFrame(array, columns=["A", "B"])

        summary = Comparer.make_summary_stat_df(df)

        assert np.isclose(summary.loc["A", "SD"], 5, rtol=0.1)
        assert np.isclose(summary.loc["B", "SD"], 1, rtol=0.1)
        assert np.isclose(summary.loc["A", "MEAN"], 100, rtol=0.1)
        assert np.isclose(summary.loc["B", "MEAN"], 25, rtol=0.1)

    def test_make_summary_stat_df_no_warning(self):
        df = pd.DataFrame({"A": [1, 2, np.nan], "B": [np.nan, 2, 3]})

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            Comparer.make_summary_stat_df(df)

    def test_process_confusion_matrix(self):
        in_cm = pd.DataFrame({1: [2, 3, 5], 2: [2, 5, 7], 3: [1, 3, 6]}, index=[1, 2, 3])

        expected_out_cm = pd.DataFrame(
            {"human 1": [2, 3, 5], "human 2": [2, 5, 7], "human 3": [1, 3, 6]},
            index=["machine 1", "machine 2", "machine 3"],
        )

        out_cm = Comparer.process_confusion_matrix(in_cm)
        assert_frame_equal(out_cm, expected_out_cm)

    def test_process_confusion_matrix_with_zero(self):
        in_cm = pd.DataFrame({0: [2, 3, 5], 1: [2, 5, 7], 2: [1, 3, 6]}, index=[0, 1, 2])

        expected_out_cm = pd.DataFrame(
            {"human 0": [2, 3, 5], "human 1": [2, 5, 7], "human 2": [1, 3, 6]},
            index=["machine 0", "machine 1", "machine 2"],
        )

        out_cm = Comparer.process_confusion_matrix(in_cm)
        assert_frame_equal(out_cm, expected_out_cm)

    def test_compute_correlations_between_versions_default_columns(self):
        df_old = pd.DataFrame(
            {
                "spkitemid": ["a", "b", "c"],
                "feature1": [1.3, 1.5, 2.1],
                "feature2": [1.1, 6.2, 2.1],
                "sc1": [2, 3, 4],
            }
        )
        df_new = pd.DataFrame(
            {
                "spkitemid": ["a", "b", "c"],
                "feature1": [-1.3, -1.5, -2.1],
                "feature2": [1.1, 6.2, 2.1],
                "sc1": [2, 3, 4],
            }
        )
        df_cors = Comparer.compute_correlations_between_versions(df_old, df_new)
        self.assertAlmostEqual(df_cors.at["feature1", "old_new"], -1.0)
        self.assertAlmostEqual(df_cors.at["feature2", "old_new"], 1.0)
        self.assertEqual(
            df_cors.at["feature1", "human_old"],
            pearsonr(df_old["feature1"], df_old["sc1"])[0],
        )
        self.assertEqual(
            df_cors.at["feature1", "human_new"],
            pearsonr(df_new["feature1"], df_new["sc1"])[0],
        )
        self.assertEqual(df_cors.at["feature1", "N"], 3)

    def test_compute_correlations_between_versions_custom_columns(self):
        df_old = pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "feature1": [1.3, 1.5, 2.1],
                "feature2": [1.1, 6.2, 2.1],
                "r1": [2, 3, 4],
            }
        )
        df_new = pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "feature1": [-1.3, -1.5, -2.1],
                "feature2": [1.1, 6.2, 2.1],
                "r1": [2, 3, 4],
            }
        )

        df_cors = Comparer.compute_correlations_between_versions(
            df_old, df_new, human_score="r1", id_column="id"
        )

        self.assertAlmostEqual(df_cors.at["feature1", "old_new"], -1.0)
        self.assertAlmostEqual(df_cors.at["feature2", "old_new"], 1.0)
        self.assertEqual(
            df_cors.at["feature1", "human_old"],
            pearsonr(df_old["feature1"], df_old["r1"])[0],
        )
        self.assertEqual(
            df_cors.at["feature1", "human_new"],
            pearsonr(df_new["feature1"], df_new["r1"])[0],
        )
        self.assertEqual(df_cors.at["feature1", "N"], 3)

    def test_compute_correlations_between_versions_no_matching_feature(self):
        df_old = pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "feature1": [1.3, 1.5, 2.1],
                "feature2": [1.1, 6.2, 2.1],
                "r1": [2, 3, 4],
            }
        )
        df_new = pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "feature3": [-1.3, -1.5, -2.1],
                "feature4": [1.1, 6.2, 2.1],
                "r1": [2, 3, 4],
            }
        )

        with self.assertRaises(ValueError):
            Comparer.compute_correlations_between_versions(
                df_old, df_new, human_score="r1", id_column="id"
            )

    def test_compute_correlations_between_versions_extra_data(self):
        df_old = pd.DataFrame(
            {
                "spkitemid": ["a", "b", "c", "d"],
                "feature1": [1.3, 1.5, 2.1, 5],
                "feature2": [1.1, 6.2, 2.1, 1],
                "feature3": [3, 5, 6, 7],
                "sc1": [2, 3, 4, 3],
            }
        )
        df_new = pd.DataFrame(
            {
                "spkitemid": ["a", "b", "c", "e"],
                "feature1": [-1.3, -1.5, -2.1, 2],
                "feature2": [1.1, 6.2, 2.1, 8],
                "feature4": [1, 3, 6, 7],
                "sc1": [2, 3, 4, 3],
            }
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            df_cors = Comparer.compute_correlations_between_versions(df_old, df_new)
        self.assertAlmostEqual(df_cors.at["feature1", "old_new"], -1.0)
        self.assertAlmostEqual(df_cors.at["feature2", "old_new"], 1.0)
        self.assertEqual(df_cors.at["feature1", "N"], 3)
        self.assertEqual(len(df_cors), 2)

    def test_compute_correlations_between_versions_no_matching_ids(self):
        df_old = pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "feature1": [1.3, 1.5, 2.1],
                "feature2": [1.1, 6.2, 2.1],
                "r1": [2, 3, 4],
            }
        )
        df_new = pd.DataFrame(
            {
                "id": ["a1", "b1", "c1"],
                "feature1": [-1.3, -1.5, -2.1],
                "feature2": [1.1, 6.2, 2.1],
                "r1": [2, 3, 4],
            }
        )

        with self.assertRaises(ValueError):
            Comparer.compute_correlations_between_versions(
                df_old, df_new, human_score="r1", id_column="id"
            )

    def test_load_rsmtool_output(self):
        source = "lr-subgroups-with-h2"
        experiment_id = "lr_subgroups_with_h2"
        test_dir = dirname(__file__)
        config_file = join(test_dir, "data", "experiments", source, f"{experiment_id}.json")
        do_run_experiment(source, experiment_id, config_file)
        output_dir = join("test_outputs", source, "output")
        figure_dir = join("test_outputs", source, "figure")

        comparer = Comparer()
        csvs, figs, file_format = comparer.load_rsmtool_output(
            output_dir, figure_dir, experiment_id, "scale", ["QUESTION", "L1"]
        )

        expected_csv_keys = [
            "df_coef",
            "df_confmatrix",
            "df_consistency",
            "df_degradation",
            "df_descriptives",
            "df_disattenuated_correlations",
            "df_disattenuated_correlations_by_L1",
            "df_disattenuated_correlations_by_L1_overview",
            "df_disattenuated_correlations_by_QUESTION",
            "df_disattenuated_correlations_by_QUESTION_overview",
            "df_eval",
            "df_eval_by_L1",
            "df_eval_by_L1_m_sd",
            "df_eval_by_L1_overview",
            "df_eval_by_QUESTION",
            "df_eval_by_QUESTION_m_sd",
            "df_eval_by_QUESTION_overview",
            "df_eval_for_degradation",
            "df_feature_cors",
            "df_mcor_sc1",
            "df_mcor_sc1_L1_overview",
            "df_mcor_sc1_QUESTION_overview",
            "df_mcor_sc1_by_L1",
            "df_mcor_sc1_by_QUESTION",
            "df_mcor_sc1_overview",
            "df_model_fit",
            "df_outliers",
            "df_pca",
            "df_pcavar",
            "df_pcor_sc1",
            "df_pcor_sc1_L1_overview",
            "df_pcor_sc1_QUESTION_overview",
            "df_pcor_sc1_by_L1",
            "df_pcor_sc1_by_QUESTION",
            "df_pcor_sc1_overview",
            "df_percentiles",
            "df_score_dist",
            "df_scores",
            "df_train_features",
            "df_true_score_eval",
        ]

        expected_fig_keys = [
            "betas",
            "eval_barplot_by_L1",
            "eval_barplot_by_QUESTION",
            "feature_boxplots_by_L1_svg",
            "feature_boxplots_by_QUESTION_svg",
            "feature_distplots",
            "pca_scree_plot",
        ]

        self.assertEqual(file_format, "csv")
        self.assertEqual(expected_csv_keys, sorted(csvs.keys()))
        self.assertEqual(expected_fig_keys, sorted(figs.keys()))
