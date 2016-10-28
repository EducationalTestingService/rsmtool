Utility Scripts
===============

In addition to the :ref:`rsmtool <usage_rsmtool>`, :ref:`rsmeval <usage_rsmeval>`, :ref:`rsmpredict <usage_rsmpredict>`, and :ref:`rsmcompare <usage_rsmpredict>` scripts, RSMTool also comes with a number of helpful utility scripts.

.. _render_notebook:

render_notebook
---------------

.. program:: render_notebook

Convert a given Jupyter notebook file (``.ipynb``) to HTML (``.html``) format.

.. option:: ipynb_file

    Path to input Jupyter notebook file.

.. option:: html_file

    Path to output HTML file.

.. option:: -h, --help

    Show help message and exit.


.. _convert_feature_json:

convert_feature_json
--------------------

.. program:: convert_feature_json

Convert an older feature JSON file to a new file in tabular format.

.. option:: --json

    Path to input feature JSON file that is to be converted.

.. option:: --output

    Path to output CSV/TSV/XLS/XLSX file containing the features in tabular format.

.. option:: --delete

    Delete original feature JSON file after conversion.

.. option:: -h, --help

    Show help message and exit.
