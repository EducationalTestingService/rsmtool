"""
Functions for formatting the output tables in Jupyter notebooks.

:author: Nitin Madnani (nmadnani@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:organization: ETS
"""

from string import Template


def float_format_func(x, prec=3):
    """
    Format float number to the specified precision

    Parameters:
    ----------
    x : float
        the float number to format

    prec: int
        the number of decimals to display

    Returns:
    -------
    ans: str
        the formatted x
    """
    formatter_string = Template('{:.${prec}f}').substitute(prec=prec)
    ans = formatter_string.format(x)
    return ans


def int_or_float_format_func(x, prec=3):
    """
    Identify whether the number if float or integer. For integer display
    no decimal, for a float round to the spedifid number of decimals.
    Parameters:
    -----------
    x : float or int
        the number to format
    prec : int
        the number of decimals to display if x is a float

    Returns:
    -------
    ans : str
        formatted x
    """
    if float.is_integer(x):
        ans = '{}'.format(int(x))
    else:
        ans = float_format_func(x, prec=prec)
    return ans


def custom_highlighter(x,
                       low=0, high=1, prec=3,
                       absolute=False,
                       span_class='bold'):
    """
    Return the supplied float as an string with specified highlighting
    using `html` tags if the value of x is below `low` or above `high`.
    Parameters:
    -----------
    x : float
        the number to format

    low : float
        x will be displayed in bold if it is below this value

    high : float
        x will be displayed in bold if it is above this value

    prec : int
        the number of decimals to display for x

    absolute: bool
        if True, use the absolute value of x for comparison

    span_class: 'bold' or 'color'
        what highlighting to use


    Returns:
    --------
    ans : str
        formatted string with `.html` tags if  highlighting is necessary

    """
    abs_x = abs(x) if absolute else x
    val = float_format_func(x, prec=prec)
    ans = '<span class="highlight_{}">{}</span>'.format(span_class, val) if abs_x < low or abs_x > high else val
    return ans


def bold_highlighter(x, low=0, high=1, prec=3, absolute=False):
    ans = custom_highlighter(x, low, high, prec, absolute, 'bold')
    return ans


def color_highlighter(x, low=0, high=1, prec=3, absolute=False):
    ans = custom_highlighter(x, low, high, prec, absolute, 'color')
    return ans

    