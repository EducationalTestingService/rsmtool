"""
Utility classes and functions for logging to Weights & Biases.

:author: Tamar Lavee (tlavee@ets.org)
"""
import wandb

EXCLUDE_WANDB_LOG = ["confMatrix"]


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


def log_configuration_to_wandb(wandb_run, configuration, name):
    """
    Log a configuration object to W&B if logging to W&B is enabled.

    Parameters
    ----------
    wandb_run : wandb.Run
        The wandb run object, or None, if logging to W&B is disabled
    configuration : rsmtool.configuration_parser.Configuration
        A Configuration object
    name : str
        Configuration name
    """
    if wandb_run:
        wandb_run.config.update({name: configuration.to_dict()})


def log_dataframe_to_wandb(wandb_run, df, df_name):
    """
    Log a dataframe as a table to W&B if logging to W&B is enabled.

    Parameters
    ----------
    wandb_run : wandb.Run
        The wandb run object, or None, if logging to W&B is disabled
    df : pandas.DataFrame
        The dataframe object
    df_name : str
        The name of the dataframe
    """
    if wandb_run and df_name not in EXCLUDE_WANDB_LOG:
        table = wandb.Table(dataframe=df, allow_mixed_types=True)
        wandb_run.log({df_name: table})


def log_report_to_wandb(wandb_run, report_name, report_path):
    """
    Log a report to W&B if logging to W&B is enabled.

    The report is logged both as an artifact and as an HTML file.

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
            wandb_run.log({report_name: wandb.Html(rf.read())})
        report_artifact = wandb.Artifact(report_name, type="html_report")
        report_artifact.add_file(local_path=report_path)
        wandb_run.log_artifact(report_artifact)
