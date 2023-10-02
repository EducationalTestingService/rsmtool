"""
Utility classes and functions for logging to Weights & Biases.

:author: Tamar Lavee (tlavee@ets.org)
"""
import wandb

# excluded dataframes will not be logged as tables or metrics.
# confusion matrices are logged separately.
EXCLUDED = ["confMatrix", "confMatrix_h1h2"]

# all values from these dataframes will be logged to the
# run as metrics in addition to logging as a table artifact.
METRICS_LOGGED = ["consistency", "eval_short", "true_score_eval"]


def init_wandb_run(config_obj):
    """
    Initialize a wandb run if logging to W&B is enabled in the configuration.

    The run object is created using the wandb project name and entity specified in
    the configuration, and full configuration is logged to this run.

    Parameters
    ----------
    config_obj : configuration_parser.Configuration
        A configuration object.

    Returns
    -------
    wandb.Run : a wandb run object, or None if logging to wandb is disabled.
    """
    use_wandb = config_obj["use_wandb"]
    wandb_run = None
    if use_wandb:
        wandb_run = wandb.init(
            project=config_obj["wandb_project"], entity=config_obj["wandb_entity"]
        )
    return wandb_run


def log_configuration_to_wandb(wandb_run, configuration):
    """
    Log a configuration object to W&B if logging to W&B is enabled.

    Parameters
    ----------
    wandb_run : wandb.Run
        The wandb run object, or None, if logging to W&B is disabled
    configuration : rsmtool.configuration_parser.Configuration
        A Configuration object
    """
    if wandb_run:
        wandb_run.config.update({configuration.context: configuration.to_dict()})


def log_dataframe_to_wandb(wandb_run, df, df_name, section=None):
    """
    Log a dataframe as a table to W&B if logging to W&B is enabled.

    Dataframes are logged as a table artifact. Values from selected
    dataframes will also be logged as metrics for easier comparison
    between runs in the W&B project dashboard.

    Parameters
    ----------
    wandb_run : wandb.Run
        The wandb run object, or None, if logging to W&B is disabled
    df : pandas.DataFrame
        The dataframe object
    df_name : str
        The name of the dataframe
    section : str
        The section in which the dataframe will we logged. If set to ``None``,
        it will be logged to the default "Charts" section. Defaults to ``None``.
    """
    if wandb_run and df_name not in EXCLUDED:
        table = wandb.Table(dataframe=df, allow_mixed_types=True)
        name = f"{section}/{df_name}" if section is not None else df_name
        wandb_run.log({name: table})
        if df_name in METRICS_LOGGED:
            indexed_df = df.set_index(df.columns[0])
            metric_dict = {}
            for column in indexed_df.columns:
                col_dict = indexed_df[column].to_dict()
                metric_dict.update(
                    {
                        get_metric_name(section, df_name, column, row): value
                        for row, value in col_dict.items()
                        if value
                    }
                )
            wandb_run.log(metric_dict)


def get_metric_name(section, df_name, col_name, row_name):
    """
    Generate the metric name for logging in W&B.

    The name contains the dataframe, column and row names,
    unless row name is empty or 0 (when dataframe has a single line)
    If a context is provided, it will be added in the beginning of
    the metric name.

    Parameters
    ----------
    section : str
        The section in which the dataframe will we logged
    df_name : str
        The dataframe name
    col_name : str
        The column name
    row_name : Union[int,str]
        The row name, or 0 for some single line dataframes.

    Returns
    -------
    metric_name : str
        The metric name
    """
    metric_name = f"{df_name}.{col_name}"
    if section is not None and section != "":
        metric_name = f"{section}/{metric_name}"
    if row_name != "" and row_name != 0:
        metric_name = f"{metric_name}.{row_name}"
    return metric_name


def log_confusion_matrix(wandb_run, human_scores, system_scores, name, section):
    """
    Log a confusion matrix to W&B if logging to W&B is enabled.

    The confusion matrix is added as a custom chart.

    Parameters
    ----------
    wandb_run : wandb.Run
        The wandb run object, or None, if logging to W&B is disabled
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


def log_report_to_wandb(wandb_run, report_name, report_path):
    """
    Log a report to W&B if logging to W&B is enabled.

    The report is logged both as an artifact and as an HTML file.
    The html is logged to a section called "reports".

    Parameters
    ----------
    wandb_run : wandb.Run
        The wandb run object, or None, if logging to W&B is disabled
    report_name: str
        The report's name
    report_path : str
        The path to the report html file.
    """
    if wandb_run:
        with open(report_path, "r") as rf:
            wandb_run.log({f"reports/{report_name}": wandb.Html(rf.read())})
        report_artifact = wandb.Artifact(report_name, type="html_report")
        report_artifact.add_file(local_path=report_path)
        wandb_run.log_artifact(report_artifact)
