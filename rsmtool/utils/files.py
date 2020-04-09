"""
Utility classes and functions for RSMTool file management.

:author: Jeremy Biggs (jbiggs@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)

:organization: ETS
"""

import json
import re

from glob import glob
from pathlib import Path
from os.path import join

from .constants import POSSIBLE_EXTENSIONS


def parse_json_with_comments(pathlike):
    """
    Parse a JSON file after removing any comments.

    Comments can use either ``//`` for single-line
    comments or or ``/* ... */`` for multi-line comments.
    The input filepath can be a string or ``pathlib.Path``.

    Parameters
    ----------
    filename : str or os.PathLike
        Path to the input JSON file either as a string
        or as a ``pathlib.Path`` object.

    Returns
    -------
    obj : dict
        JSON object representing the input file.

    Note
    ----
    This code was adapted from:
    https://web.archive.org/web/20150520154859/http://www.lifl.fr/~riquetd/parse-a-json-file-with-comments.html
    """

    # Regular expression to identify comments
    comment_re = re.compile(r'(^)?[^\S\n]*/(?:\*(.*?)\*/[^\S\n]*|/[^\n]*)($)?',
                            re.DOTALL | re.MULTILINE)

    # if we passed in a string, convert it to a Path
    if isinstance(pathlike, str):
        pathlike = Path(pathlike)

    with open(pathlike, 'r') as file_buff:
        content = ''.join(file_buff.readlines())

        # Looking for comments
        match = comment_re.search(content)
        while match:

            # single line comment
            content = content[:match.start()] + content[match.end():]
            match = comment_re.search(content)

        # Return JSON object
        config = json.loads(content)
        return config


def has_files_with_extension(directory, ext):
    """
    Check if the directory has any files with the given extension.

    Parameters
    ----------
    directory : str
        The path to the directory where output is located.
    ext : str
        The the given extension.

    Returns
    -------
    bool
        True if directory contains files with given extension,
        else False.
    """
    files_with_extension = glob(join(directory, '*.{}'.format(ext)))
    return len(files_with_extension) > 0


def get_output_directory_extension(directory, experiment_id):
    """
    Check the output directory to determine what file extensions
    exist. If more than one extension (in the possible list of
    extensions) exists, then raise a ValueError. Otherwise,
    return the one file extension. If no extensions can be found, then
    `csv` will be returned by default.

    Possible extensions include: `csv`, `tsv`, `xlsx`. Files in the
    directory with none of these extensions will be ignored.

    Parameters
    ----------
    directory : str
        The path to the directory where output is located.
    experiment_id : str
        The ID of the experiment.

    Returns
    -------
    extension : {'csv', 'tsv', 'xlsx'}
        The extension that output files in this directory
        end with.

    Raises
    ------
    ValueError
        If any files in the directory have different extensions,
        and are in the list of possible output extensions.
    """
    extension = 'csv'
    extensions_identified = {ext for ext in POSSIBLE_EXTENSIONS
                             if has_files_with_extension(directory, ext)}

    if len(extensions_identified) > 1:
        raise ValueError('Some of the files in the experiment output directory (`{}`) '
                         'for `{}` have different extensions. All files in this directory '
                         'must have the same extension. The following extensions were '
                         'identified : {}'.format(directory,
                                                  experiment_id,
                                                  ', '.join(extensions_identified)))

    elif len(extensions_identified) == 1:
        extension = list(extensions_identified)[0]

    return extension
