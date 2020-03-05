"""
Utility classes and functions for type conversion.

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

from skll.data import safe_float as string_to_number


def int_to_float(value):
    """
    Convert integer to float, if possible.

    Parameters
    ----------
    value
        Name of the experiment file we want to locate.

    Returns
    -------
    value
        Value converted to float, if possible
    """
    return float(value) if type(value) == int else value


def convert_to_float(value):
    """
    Convert value to float, if possible.

    Parameters
    ----------
    value
        Name of the experiment file we want to locate.

    Returns
    -------
    value
        Value converted to float, if possible
    """
    return int_to_float(string_to_number(value))
