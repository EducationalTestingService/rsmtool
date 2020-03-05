"""
Utility functions for use in RSMTool sections/notebooks.

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

from math import ceil
from os.path import exists, isabs, relpath
from pathlib import Path
from string import Template
from textwrap import wrap
from IPython.display import display, HTML


HTML_STRING = ("""<li><b>{}</b>: <a href="{}" download>{}</a></li>""")


def float_format_func(num, prec=3):
    """
    Format the given floating point number to the specified precision
    and return as a string.

    Parameters:
    ----------
    num : float
        The floating point number to format.
    prec: int, optional
        The number of decimal places to use when displaying the number.
        Defaults to 3.

    Returns:
    -------
    ans: str
        The formatted string representing the given number.
    """

    formatter_string = Template('{:.${prec}f}').substitute(prec=prec)
    ans = formatter_string.format(num)
    return ans


def int_or_float_format_func(num, prec=3):
    """
    Identify whether the number is float or integer. When displaying
    integers, use no decimal. For a float, round to the specified
    number of decimal places. Return as a string.

    Parameters:
    -----------
    num : float or int
        The number to format and display.
    prec : int, optional
        The number of decimal places to display if x is a float.
        Defaults to 3.

    Returns:
    -------
    ans : str
        The formatted string representing the given number.
    """

    if float.is_integer(num):
        ans = '{}'.format(int(num))
    else:
        ans = float_format_func(num, prec=prec)
    return ans


def custom_highlighter(num,
                       low=0,
                       high=1,
                       prec=3,
                       absolute=False,
                       span_class='bold'):
    """
    Return the supplied float as an HTML <span> element with the specified
    class if its value is below ``low`` or above ``high``. If its value does
    not meet those constraints, then return as a plain string with the
    specified number of decimal places.

    Parameters:
    -----------
    num : float
        The floating point number to format.
    low : float
        The number will be displayed as an HTML span it is below this value.
        Defaults to 0.
    high : float
        The number will be displayed as an HTML span it is above this value.
        Defaults to 1.
    prec : int
        The number of decimal places to display for x. Defaults to 3.
    absolute: bool
        If True, use the absolute value of x for comparison.
        Defaults to False.
    span_class: str
        One of ``bold`` or ``color``. These are the two classes
        available for the HTML span tag.

    Returns:
    --------
    ans : str
        The formatted (plain or HTML) string representing the given number.
    """
    abs_num = abs(num) if absolute else num
    val = float_format_func(num, prec=prec)
    ans = ('<span class="highlight_{}">{}</span>'.format(span_class, val)
           if abs_num < low or abs_num > high else val)
    return ans


def bold_highlighter(num, low=0, high=1, prec=3, absolute=False):
    """
    Instantiating ``custom_highlighter()`` with the ``bold`` class as
    the default.

    Parameters:
    -----------
    num : float
        The floating point number to format.
    low : float
        The number will be displayed as an HTML span it is below this value.
        Defaults to 0.
    high : float
        The number will be displayed as an HTML span it is above this value.
        Defaults to 1.
    prec : int
        The number of decimal places to display for x.
        Defaults to 3.
    absolute: bool
        If True, use the absolute value of x for comparison.
        Defaults to False.

    Returns:
    --------
    ans : str
        The formatted highlighter with bold class as default.
    """
    ans = custom_highlighter(num, low, high, prec, absolute, 'bold')
    return ans


def color_highlighter(num, low=0, high=1, prec=3, absolute=False):
    """
    Instantiating ``custom_highlighter()`` with the ``color`` class as
    the default.

    Parameters:
    -----------
    num : float
        The floating point number to format.
    low : float
        The number will be displayed as an HTML span it is below this value.
        Defaults to 0.
    high : float
        The number will be displayed as an HTML span it is above this value.
        Defaults to 1.
    prec : int
        The number of decimal places to display for x.
        Defaults to 3.
    absolute: bool
        If True, use the absolute value of x for comparison.
        Defaults to False.

    Returns:
    --------
    ans : str
        The formatted highlighter with color class as default.
    """
    ans = custom_highlighter(num, low, high, prec, absolute, 'color')
    return ans


def compute_subgroup_plot_params(group_names,
                                 num_plots):
    """
    Computing subgroup plot and figure parameters based on number of
    subgroups and number of plots to be generated.

    Parameters
    ----------
    group_names : list
        A list of subgroup names for plots.
    num_plots : int
        The number of plots to compute.

    Returns
    -------
    figure_width : int
        The width of the figure.
    figure_height : int
        The height of the figure.
    num_rows : int
        The number of rows for the plots.
    num_columns : int
        The number of columns for the plots.
    wrapped_group_names : list of str
        A list of group names for plots.
    """
    wrapped_group_names = ['\n'.join(wrap(str(gn), 20)) for gn in group_names]
    plot_height = 4 if wrapped_group_names == group_names else 6
    num_groups = len(group_names)
    if num_groups <= 6:
        num_columns = 2
        num_rows = ceil(num_plots / num_columns)
        figure_width = num_columns * num_groups
        figure_height = plot_height * num_rows
    else:
        num_columns = 1
        num_rows = num_plots
        figure_width = 10
        figure_height = plot_height * num_plots

    return (figure_width, figure_height, num_rows, num_columns, wrapped_group_names)


def get_thumbnail_as_html(path_to_image, image_id, path_to_thumbnail=None):
    """
    Given an path to an image file, generate the HTML for
    a click-able thumbnail version of the image.
    On click, this HTML will open the full-sized version
    of the image in a new window.

    Parameters
    ----------
    path_to_image : str
        The absolute or relative path to the image.
        If an absolute path is provided, it will be
        converted to a relative path.
    image_id : int
        The id of the <img> tag in the HTML. This must
        be unique for each <img> tag.
    path_to_thumbnail : str or None, optional
        If you would like to use a different thumbnail
        image, specify the path to this thumbnail.
        Defaults to None.

    Returns
    -------
    image : str
        The HTML string generated for the image.

    Raises
    ------
    FileNotFoundError
        If the image file cannot be located.
    """
    error_message = 'The file `{}` could not be located.'
    if not exists(path_to_image):
        raise FileNotFoundError(error_message.format(path_to_image))

    # check if the path is relative or absolute
    if isabs(path_to_image):
        rel_image_path = relpath(path_to_image)
    else:
        rel_image_path = path_to_image

    # if `path_to_thumbnail` is None, use `path_to_image`;
    # otherwise, get the relative path to the thumbnail
    if path_to_thumbnail is None:
        rel_thumbnail_path = rel_image_path
    else:
        if not exists(path_to_thumbnail):
            raise FileNotFoundError(error_message.format(path_to_thumbnail))
        rel_thumbnail_path = relpath(path_to_thumbnail)

    # specify the thumbnail style
    style = """
    <style>
    img {
        border: 1px solid #ddd;
        border-radius: 4px;
        padding: 5px;
        width: 150px;
        cursor: pointer;
    }
    </style>
    """

    # on click, open larger image in new window
    script = """
    <script>
    function getPicture(picpath) {{
        window.open(picpath, 'Image', resizable=1);
    }};
    </script>""".format(image_id)

    # generate image tags
    image = ("""<img id='{}' src='{}' onclick='getPicture("{}")' """
             """title="Click to enlarge">"""
             """</img>""").format(image_id,
                                  rel_image_path,
                                  rel_thumbnail_path)

    # create the image HTML
    image += style
    image += script
    return image


def show_thumbnail(path_to_image, image_id, path_to_thumbnail=None):
    """
    Given an path to an image file, display
    a click-able thumbnail version of the image.
    On click, open the full-sized version of the
    image in a new window.

    Parameters
    ----------
    path_to_image : str
        The absolute or relative path to the image.
        If an absolute path is provided, it will be
        converted to a relative path.
    image_id : int
        The id of the <img> tag in the HTML. This must
        be unique for each <img> tag.
    path_to_thumbnail : str or None, optional
        If you would like to use a different thumbnail
        image, specify the path to the thumbnail.
        Defaults to None.

    Displays
    --------
    display : IPython.core.display.HTML
        The HTML display of the thumbnail image.
    """
    display(HTML(get_thumbnail_as_html(path_to_image,
                                       image_id,
                                       path_to_thumbnail)))


def get_files_as_html(output_dir, experiment_id, file_format, replace_dict={}):
    """
    Generate HTML list items for each file name,
    given output directory. Optionally pass a
    replacement dictionary to use more descriptive
    titles for the file names.

    Parameters
    ----------
    output_dir : str
        The output directory.
    experiment_id : str
        The experiment ID.
    file_format : str
        The format of the output files.
    replace_dict : dict, optional
        A dictionary which makes file names to descriptions.
        Defaults to empty dictionary.

    Returns
    ------
    html_string : str
        HTML string with file descriptions and links.
    """
    output_dir = Path(output_dir)
    parent_dir = output_dir.parent
    files = output_dir.glob('*.{}'.format(file_format))
    html_string = ''
    for file in sorted(files):
        relative_file = ".." / file.relative_to(parent_dir)
        relative_name = relative_file.stem.replace('{}_'.format(experiment_id), '')

        # check if relative name is in the replacement dictionary and,
        # if it is, use the more descriptive name in the replacement
        # dictionary. Otherwise, normalize the file name and use that
        # as the description instead.
        if relative_name in replace_dict:
            descriptive_name = replace_dict[relative_name]
        else:
            descriptive_name_components = relative_name.split('_')
            descriptive_name = ' '.join(descriptive_name_components).title()

        html_string += HTML_STRING.format(descriptive_name,
                                          relative_file,
                                          file_format)

    return """<ul><html>""" + html_string + """</ul></html>"""


def show_files(output_dir, experiment_id, file_format, replace_dict={}):
    """
    Show files for a given output directory.

    Parameters
    ----------
    output_dir : str
        The output directory.
    experiment_id : str
        The experiment ID.
    file_format : str
        The format of the output files.
    replace_dict : dict, optional
        A dictionary which makes file names to descriptions.
        Defaults to empty dictionary.

    Displays
    --------
    display : IPython.core.display.HTML
        The HTML file descriptions and links.
    """
    html_string = get_files_as_html(output_dir,
                                    experiment_id,
                                    file_format,
                                    replace_dict)
    display(HTML(html_string))
