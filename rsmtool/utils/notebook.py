"""
Utility functions for use in RSMTool sections/notebooks.

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

import re
from math import ceil
from os.path import exists, isabs, relpath
from pathlib import Path
from string import Template
from textwrap import wrap
from typing import Dict, List, Optional, Tuple, Union

from IPython.display import HTML, display

from .constants import INTERMEDIATE_FILES_TO_DESCRIPTIONS

INTERMEDIATE_TABLE_ROW_STRING = """
<tr>
    <td style="text-align:left;">
        <a href="{}" style="text-decoration: none;" download>{}</a>
    </td>
    <td style="text-align:left;">{}</td>
</tr>
"""


def float_format_func(num: float, prec: int = 3, scientific: bool = False) -> str:
    """
    Format given float to the specified precision as a string.

    Parameters
    ----------
    num : float
        The floating point number to format.
    prec: int
        The number of decimal places to use when displaying the number.
        Defaults to 3.
    scientific: bool
        Whether to display the number in scientific notiation if the rounded
        version is "0.000".
        Defaults to ``False``.

    Returns
    -------
    ans : str
        The formatted string representing the given number.
    """
    rounded_formatter_string = Template("{:.${prec}f}").substitute(prec=prec)
    scientific_formatter_string = Template("{:.${prec}e}").substitute(prec=prec)
    ans = rounded_formatter_string.format(num)
    if ans == "0.000" and scientific:
        ans = scientific_formatter_string.format(num)
    return ans


def int_or_float_format_func(num: Union[int, float], prec: int = 3) -> str:
    """
    Identify whether the number is float or integer.

    When displaying integers, use no decimal. For a float, round to the
    specified number of decimal places and convert to a string.

    Parameters
    ----------
    num : Union[int, float]
        The number to format and display.
    prec : int
        The number of decimal places to display if x is a float.
        Defaults to 3.

    Returns
    -------
    ans : str
        The formatted string representing the given number.
    """
    if float.is_integer(num):
        ans = f"{int(num)}"
    else:
        ans = float_format_func(num, prec=prec)
    return ans


def custom_highlighter(
    num: float,
    low: float = 0,
    high: float = 1,
    prec: int = 3,
    absolute: bool = False,
    span_class: str = "bold",
) -> str:
    """
    Convert float to an HTML <span> element with given class.

    The conversion only happens if the given float is below ``low``
    or above ``high``. If it does not meet those constraints, then
    a plain string is returned with specified number of decimal places.

    Parameters
    ----------
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
    absolute : bool
        If ``True``, use the absolute value of x for comparison.
        Defaults to ``False``.
    span_class : str
        One of ``"bold"`` or ``"color"``. These are the two classes available
        for the HTML span tag.
        Defaults to ``"bold"``.

    Returns
    -------
    ans : str
        The plain or HTML string representing the given number.
    """
    abs_num = abs(num) if absolute else num
    val = float_format_func(num, prec=prec)
    ans = (
        f'<span class="highlight_{span_class}">{val}</span>'
        if abs_num < low or abs_num > high
        else val
    )
    return ans


def bold_highlighter(
    num: float, low: float = 0, high: float = 1, prec: int = 3, absolute: bool = False
) -> str:
    """
    Instantiate a ``custom_highlighter()`` with "bold" as default class.

    Parameters
    ----------
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
    absolute : bool
        If ``True``, use the absolute value of x for comparison.
        Defaults to ``False``.

    Returns
    -------
    ans : str
        The formatted highlighter with bold class as default.
    """
    ans = custom_highlighter(num, low, high, prec, absolute, "bold")
    return ans


def color_highlighter(
    num: float, low: float = 0, high: float = 1, prec: int = 3, absolute: bool = False
) -> str:
    """
    Instantiate a ``custom_highlighter()`` with "color" as the default class.

    Parameters
    ----------
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
    absolute : bool
        If ``True``, use the absolute value of x for comparison.
        Defaults to ``False``.

    Returns
    -------
    ans : str
        The formatted highlighter with color class as default.
    """
    ans = custom_highlighter(num, low, high, prec, absolute, "color")
    return ans


def compute_subgroup_plot_params(
    group_names: List[str], num_plots: int
) -> Tuple[int, int, int, int, List[str]]:
    """
    Compute subgroup plot and figure parameters.

    The parameters are computed based on the number of subgroups and the
    number of plots to be generated.

    Parameters
    ----------
    group_names : List[str]
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
    wrapped_group_names : List[str]
        A list of group names for plots.
    """
    wrapped_group_names = ["\n".join(wrap(str(gn), 20)) for gn in group_names]
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


def get_thumbnail_as_html(
    path_to_image: str, image_id: int, path_to_thumbnail: Optional[str] = None
) -> str:
    """
    Generate HTML for a clickable thumbnail of given image.

    Given the path to an image file, generate the HTML for
    a clickable thumbnail version of the image. When clicked,
    this HTML will open the full-sized version of the image in
    a new window.

    Parameters
    ----------
    path_to_image : str
        The absolute or relative path to the image. If an absolute path is
        provided, it will be converted to a relative path.
    image_id : int
        The id of the <img> tag in the HTML. This must be unique for each <img> tag.
    path_to_thumbnail : Optional[str]
        If you would like to use a different thumbnail image, specify the path
        to this thumbnail.
        Defaults to ``None``.

    Returns
    -------
    image : str
        The HTML string generated for the image.

    Raises
    ------
    FileNotFoundError
        If the image file cannot be located.
    """
    error_message = "The file `{}` could not be located."
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
    function getPicture(picpath) {
        window.open(picpath, 'Image', resizable=1);
    };
    </script>"""

    # generate image tags
    image = (
        f"""<img id='{image_id}' src='{rel_image_path}' """
        f"""onclick='getPicture("{rel_thumbnail_path}")' """
        f"""title="Click to enlarge"></img>"""
    )

    # create the image HTML
    image += style
    image += script
    return image


def show_thumbnail(
    path_to_image: str, image_id: int, path_to_thumbnail: Optional[str] = None
) -> None:
    """
    Display the HTML for an image thumbnail in a Jupyter notebook.

    Given the path to an image file, generate the HTML for its
    thumbnail and display it in the notebook.

    Parameters
    ----------
    path_to_image : str
        The absolute or relative path to the image. If an absolute path is
        provided, it will be converted to a relative path.
    image_id : int
        The id of the <img> tag in the HTML. This must be unique for each <img> tag.
    path_to_thumbnail : Optional[str]
        If you would like to use a different thumbnail image, specify the path
        to the thumbnail.
        Defaults to ``None``.
    """
    display(HTML(get_thumbnail_as_html(path_to_image, image_id, path_to_thumbnail)))


def get_files_as_html(
    output_dir: str, experiment_id: str, file_format: str, replace_dict: Dict[str, str] = {}
) -> str:
    """
    Generate an HTML list for each output file in given directory.

    Optionally pass a replacement dictionary to use more descriptive
    titles for the file names.

    Parameters
    ----------
    output_dir : str
        The output directory.
    experiment_id : str
        The experiment ID.
    file_format : str
        The format of the output files.
    replace_dict : Dict[str, str]
        A dictionary which maps file names to descriptions.
        Defaults to ``{}``.

    Returns
    -------
    html_string : str
        HTML string with file descriptions and links.
    """
    output_path = Path(output_dir)
    parent_dir = output_path.parent

    # get the list of intermediate files generated for this experiment
    files = list(output_path.glob(f"*.{file_format}"))

    # get the bare filenames for each file since we will need those for sorting
    filenames = [file.stem.replace(f"{experiment_id}_", "") for file in files]

    # initialize list to hold the generated HTML strings
    html_strings = [
        """
        <tr>
            <th style="text-align:left;">Filename</th>
            <th style="text-align:left;">Description</th>
        </tr>
        """
    ]

    # iterate over the files and generate the HTML strings
    for filename, file in sorted(zip(filenames, files)):
        # get the path of the file relative to the parent of the "output"
        # directory; these will be the targets of our links
        relative_file = ".." / file.relative_to(parent_dir)

        # get the relative names of the file without the experiment id prefix
        # these are needed in case we do not have a pre-defined replacement
        relative_name = relative_file.stem.replace(f"{experiment_id}_", "")

        # if the name of the file contains "_by_<subgroup>", replace
        # the subgroup with "ZZZ" to get the replacement key; otherwise,
        # use the filename as the key; if neither form of the  key is
        # found in the dictionary, use the filename itself as the description
        try:
            if m := re.search(r"_by_([^\._]+)", filename):
                subgroup = m.group(1)
                replacement_key = filename.replace(f"_by_{subgroup}", "_by_ZZZ")
                # replace the "ZZZ" with the actual subgroup in the description
                descriptive_name = replace_dict[replacement_key].replace("ZZZ", subgroup)
            else:
                replacement_key = filename
                descriptive_name = replace_dict[filename]
        except KeyError:
            descriptive_name_components = relative_name.split("_")
            descriptive_name = " ".join(descriptive_name_components).title()

        # generate a table row string for this file
        html_strings.append(
            INTERMEDIATE_TABLE_ROW_STRING.format(relative_file, filename, descriptive_name)
        )

    # combine the HTML table row strings into a single HTML table
    table_rows = "\n".join(html_strings)
    return f"<table>{table_rows}</table>"


def show_files(context: str, output_dir: str, experiment_id: str, file_format: str) -> None:
    """
    Show files for given context and directory as a table in a Jupyter notebook.

    Parameters
    ----------
    context: str
        The tool context: one of {``"rsmtool"``, ``"rsmeval"``, ``"rsmsummarize"``}.
    output_dir : str
        The output directory.
    experiment_id : str
        The experiment ID.
    file_format : str
        The format of the output files.
    """
    html_string = get_files_as_html(
        output_dir,
        experiment_id,
        file_format,
        replace_dict=INTERMEDIATE_FILES_TO_DESCRIPTIONS[context],
    )
    display(HTML(html_string))
