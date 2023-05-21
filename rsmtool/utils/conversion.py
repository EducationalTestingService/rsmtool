"""
Utility classes and functions for type conversion.

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

from skll.data import safe_float as string_to_number

from .constants import RSMEXPLAIN_RANGE_REGEXP


def int_to_float(value):
    """
    Convert integer to float, if possible.

    Parameters
    ----------
    value
        Name of the value we want to convert.

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


def parse_range(value):
    """
    Parse range strings used in rsmexplain.

    Valid range strings are two positive integers separated by a hyphen ("10-20").

    Parameters
    ----------
    value : str
        The range string we want to parse.

    Returns
    -------
    Tuple[int, int]
        A tuple containing two integers.
    """
    valid_range = False
    if matchobj := RSMEXPLAIN_RANGE_REGEXP.match(value):
        groups = matchobj.groupdict()
        start_index = string_to_number(groups.get("start"))
        end_index = string_to_number(groups.get("end"))

        # the range is valid if and only if:
        # - both start and end indices are positive integers
        # - and start index < end index
        valid_range = (
            isinstance(start_index, int)
            and isinstance(end_index, int)
            and 0 <= start_index < end_index
        )

    if not valid_range:
        raise ValueError(f"invalid value '{value}' specified for range")
    else:
        return [start_index, end_index]
