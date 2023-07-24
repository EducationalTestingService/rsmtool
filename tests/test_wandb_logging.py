import unittest
from os.path import join
from unittest.mock import Mock, patch

import pandas as pd

from rsmtool.test_utils import rsmtool_test_dir
from rsmtool.utils.wandb import (
    init_wandb_run,
    log_dataframe_to_wandb,
    log_report_to_wandb,
)


class TestWandbLogging(unittest.TestCase):
    def test_init_wandb_run_wandb_enabled(self):
        config = {"use_wandb": True, "wandb_entity": "test_entity", "wandb_project": "test_project"}

        mock_wandb_run = Mock()
        with patch("wandb.init") as mock_wandb_init:
            mock_wandb_init.return_value = mock_wandb_run
            init_wandb_run(config)
        mock_wandb_init.assert_called_with(project="test_project", entity="test_entity")
        mock_wandb_run.config.update.assert_called_with(config)

    def test_init_wandb_run_wandb_disabled(self):
        config = {
            "use_wandb": False,
            "wandb_entity": "test_entity",
            "wandb_project": "test_project",
        }

        self.assertIsNone(init_wandb_run(config))

    def test_log_dataframe_to_wandb_enabled(self):
        df = pd.DataFrame(
            {
                "X1": [1.3, 1.2, 1.5, 1.7, 1.8, 1.9, 2.0],
                "X2": [1.3, 1.2, 1.5, 1.7001, 1.8, 1.9, 2.0],
            }
        )
        with patch("wandb.Table") as mock_wandb_table:
            mock_wandb_run = Mock()
            log_dataframe_to_wandb(mock_wandb_run, df, "df_name")
            mock_wandb_table.assert_called_with(dataframe=df, allow_mixed_types=True)
            mock_wandb_run.log.assert_called_once()

    def test_log_dataframe_to_wandb_enabled_excluded_table(self):
        df = pd.DataFrame(
            {
                "X1": [1.3, 1.2, 1.5, 1.7, 1.8, 1.9, 2.0],
                "X2": [1.3, 1.2, 1.5, 1.7001, 1.8, 1.9, 2.0],
            }
        )
        with patch("wandb.Table") as mock_wandb_table:
            mock_wandb_run = Mock()
            log_dataframe_to_wandb(mock_wandb_run, df, "confMatrix")
            mock_wandb_table.assert_not_called()
            mock_wandb_run.log.assert_not_called()

    def test_log_dataframe_to_wandb_disabled(self):
        df = pd.DataFrame(
            {
                "X1": [1.3, 1.2, 1.5, 1.7, 1.8, 1.9, 2.0],
                "X2": [1.3, 1.2, 1.5, 1.7001, 1.8, 1.9, 2.0],
            }
        )
        with patch("wandb.Table") as mock_wandb_table:
            wandb_run = None
            log_dataframe_to_wandb(wandb_run, df, "df_name")
            mock_wandb_table.assert_not_called()

    def test_log_report_to_wandb_enabled(self):
        report_path = join(rsmtool_test_dir, "data", "files", "wandb_test_file.html")
        with patch("wandb.Artifact") as mock_wandb_artifact:
            # create mock objects
            mock_wandb_run = Mock()
            mock_report_artifact = Mock()
            mock_wandb_artifact.return_value = mock_report_artifact

            log_report_to_wandb(mock_wandb_run, report_path)
            # assert that all calls have been made
            mock_wandb_artifact.assert_called_once()
            mock_report_artifact.add_file.assert_called_with(local_path=report_path)
            mock_wandb_run.log.assert_called_once()
            mock_wandb_run.log_artifact.assert_called_once()

    def test_log_report_to_wandb_disabled(self):
        report_path = join(rsmtool_test_dir, "data", "files", "wandb_test_file.html")
        with patch("wandb.Artifact") as mock_wandb_artifact:
            # create mock objects, wandb_run is None when wandb is disabled
            wandb_run = None
            mock_report_artifact = Mock()
            mock_wandb_artifact.return_value = mock_report_artifact

            log_report_to_wandb(wandb_run, report_path)
            # assert that no calls to wandb objects have been made
            mock_wandb_artifact.assert_not_called()
            mock_report_artifact.add_file.assert_not_called()
