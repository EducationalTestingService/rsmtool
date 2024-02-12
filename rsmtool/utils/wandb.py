"""
Utility classes and functions for logging to Weights & Biases.

:author: Tamar Lavee (tlavee@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
"""

from typing import Optional, Union

import pandas as pd
import wandb
from wandb.sdk.lib import RunDisabled
from wandb.wandb_run import Run

from ..configuration_parser import Configuration

# excluded dataframes will not be logged as tables or metrics.
# confusion matrices are logged separately.
EXCLUDED = ["confMatrix", "confMatrix_h1h2"]

# all values from these dataframes will be logged to the
# run as metrics in addition to logging as a table artifact.
METRICS_LOGGED = ["consistency", "eval_short", "true_score_eval"]


def init_wandb_run(config_obj: Configuration) -> Union[Run, RunDisabled, None]:
    """
    Initialize a wandb run if logging to wandb is enabled in the configuration.

    The Run object is created using the wandb project name and entity specified
    in the configuration, and the full configuration is logged to this run.

    Parameters
    ----------
    config_obj : Configuration
        The configuration object containing the wandb project name and entity.

    Returns
    -------
    Union[wandb.wandb_run.Run, wandb.sdk.lib.RunDisabled, None]
        A wandb Run object, or ``None`` if logging to wandb is disabled.
    """
    use_wandb = config_obj["use_wandb"]
    wandb_run = None
    if use_wandb:
        wandb_run = wandb.init(
            project=config_obj["wandb_project"], entity=config_obj["wandb_entity"]
        )
    return wandb_run


def log_configuration_to_wandb(
    wandb_run: Union[Run, RunDisabled, None], configuration: Configuration
) -> None:
    """
    Log a configuration object to wandb if logging to wandb is enabled.

    Parameters
    ----------
    wandb_run : Union[wandb.wandb_run.Run, wandb.sdk.lib.RunDisabled, None]
        The wandb Run object, or ``None``, if logging to wandb is disabled.
    configuration : rsmtool.configuration_parser.Configuration
        The Configuration object to log to the run.
    """
    if wandb_run:
        wandb_run.config.update({configuration.context: configuration.to_dict()})


def log_dataframe_to_wandb(
    wandb_run: Union[Run, RunDisabled, None],
    df: pd.DataFrame,
    frame_name: pd.DataFrame,
    section: Optional[str] = None,
) -> None:
    """
    Log a dataframe as a table to wandb if logging to wandb is enabled.

    Dataframes are logged as a table artifact. Values from selected
    dataframes will also be logged as metrics for easier comparison
    between runs in the wandb project dashboard.

    Parameters
    ----------
    wandb_run : Union[wandb.wandb_run.Run, wandb.sdk.lib.RunDisabled, None]
        The wandb Run object, or ``None``, if logging to wandb is disabled.
    df : pandas.DataFrame
        The dataframe object to log.
    frame_name : str
        The name of the dataframe to use in the log.
    section : str
        The section in which the dataframe will we logged. If set to ``None``,
        it will be logged to the default "Charts" section.
        Defaults to ``None``.
    """
    if wandb_run and frame_name not in EXCLUDED:
        table = wandb.Table(dataframe=df, allow_mixed_types=True)
        name = f"{section}/{frame_name}" if section is not None else frame_name
        wandb_run.log({name: table})
        if frame_name in METRICS_LOGGED:
            indexed_df = df.set_index(df.columns[0])
            metric_dict = {}
            for column in indexed_df.columns:
                col_dict = indexed_df[column].to_dict()
                metric_dict.update(
                    {
                        get_metric_name(section, frame_name, column, row): value
                        for row, value in col_dict.items()
                        if value
                    }
                )
            wandb_run.log(metric_dict)


def get_metric_name(
    section: Optional[str], frame_name: str, col_name: str, row_name: Union[int, str]
) -> str:
    """
    Generate the metric name for logging in wandb.

    The name contains the dataframe, column and row names,
    unless row name is empty or 0 (when dataframe has a single line)
    If a context is provided, it will be added in the beginning of
    the metric name.

    Parameters
    ----------
    section : Optional[str]
        The section name to use in the log. If set to ``None``,
        it will not be added to the metric name.
    frame_name : str
        The dataframe name to use in the log.
    col_name : str
        The column name to use in the log.
    row_name : Union[int, str]
        The row name, or 0 for some single line dataframes.

    Returns
    -------
    metric_name : str
        The metric name to be logged.
    """
    metric_name = f"{frame_name}.{col_name}"
    if section is not None and section != "":
        metric_name = f"{section}/{metric_name}"
    if row_name != "" and row_name != 0:
        metric_name = f"{metric_name}.{row_name}"
    return metric_name


def log_confusion_matrix(
    wandb_run: Union[Run, RunDisabled, None],
    human_scores: pd.Series,
    system_scores: pd.Series,
    name: str,
    section: str,
) -> None:
    """
    Log a confusion matrix to wandb if logging to wandb is enabled.

    The confusion matrix is added as a custom chart.

    Parameters
    ----------
    wandb_run : Union[wandb.wandb_run.Run, wandb.sdk.lib.RunDisabled, None]
        The wandb Run object, or ``None``, if logging to wandb is disabled.
    human_scores : pandas.Series
        The human scores for the responses in the data
    system_scores : pandas.Series
        The predicted scores for the responses in the data
    name : str
        The chart title
    section : str
        The section in which the confusion matrix will we logged
    """
    if wandb_run:
        wandb_run.log(
            {
                f"{section}/{name}": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=human_scores.values,
                    preds=system_scores.values,
                    title=name,
                )
            }
        )


def log_report_to_wandb(
    wandb_run: Union[Run, RunDisabled, None], report_name: str, report_path: str
):
    """
    Log a report to wandb if logging to wandb is enabled.

    The report is logged both as an artifact and as an HTML file.
    The html is logged to a section called "reports".

    Parameters
    ----------
    wandb_run : Union[wandb.wandb_run.Run, wandb.sdk.lib.RunDisabled, None]
        The wandb Run object, or ``None``, if logging to wandb is disabled.
    report_name: str
        The report's name to use in the log.
    report_path : str
        The path to the report html file that is to be logged.
    """
    if wandb_run:
        with open(report_path, "r") as rf:
            wandb_run.log({f"reports/{report_name}": wandb.Html(rf.read())})
        report_artifact = wandb.Artifact(report_name, type="html_report")
        report_artifact.add_file(local_path=report_path)
        wandb_run.log_artifact(report_artifact)
