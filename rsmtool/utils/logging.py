"""
Utility classes and functions for RSMTool logging.

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

import contextlib
import logging

import joblib


class LogFormatter(logging.Formatter):
    """
    Custom logging formatter.

    Note
    ----
    This class is adapted from https://stackoverflow.com/q/1343227.
    """

    info_fmt = "%(msg)s"
    warn_fmt = "WARNING: %(msg)s"

    err_fmt = "ERROR: %(msg)s"
    dbg_fmt = "DEBUG: %(module)s: %(lineno)d: %(msg)s"

    def __init__(self, fmt="%(levelno)s: %(msg)s"):  # noqa: D107
        logging.Formatter.__init__(self, fmt)

    def format(self, record):
        """
        Format the given record.

        Parameters
        ----------
        record : logging.LogRecord
            The record to format.
        """
        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._fmt

        # Replace the original format with one customized by logging level
        if record.levelno == logging.DEBUG:
            self._fmt = LogFormatter.dbg_fmt
            self._style = logging.PercentStyle(self._fmt)

        elif record.levelno == logging.WARNING:
            self._fmt = LogFormatter.warn_fmt
            self._style = logging.PercentStyle(self._fmt)

        elif record.levelno == logging.INFO:
            self._fmt = LogFormatter.info_fmt
            self._style = logging.PercentStyle(self._fmt)

        elif record.levelno == logging.ERROR:
            self._fmt = LogFormatter.err_fmt
            self._style = logging.PercentStyle(self._fmt)

        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        self._fmt = format_orig

        return result


def get_file_logger(logger_name, log_file_path):
    """
    Create and return a file-based logger.

    Instantiate a logger with the given name and attach a
    file handler that uses the given log file. The logging
    level is set to INFO.

    Parameters
    ----------
    logger_name : str
        Name to use for the logger.
    log_file_path : str
        File path to which the logger should output messages.

    Returns
    -------
    logger
        A logging.Logger object attached to a file handler.
    """
    logger = logging.getLogger(logger_name)
    logger.propagate = False
    formatter = logging.Formatter("%(message)s")
    file_handler = logging.FileHandler(log_file_path, mode="w")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    return logger


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Patch joblib to report into tqdm progress bar given as argument.

    This function creates a joblib-compatible context manager with
    a progress bar attached.

    Adapted from: https://stackoverflow.com/a/58936697

    Parameters
    ----------
    tqdm_object : tqdm progress bar
        The given tqdm progress bar into which joblib should report
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
