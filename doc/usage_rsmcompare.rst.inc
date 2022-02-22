.. _usage_rsmcompare:

``rsmcompare`` - Create a detailed comparison of two scoring models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RSMTool provides the ``rsmcompare`` command-line utility to compare two models and to generate a detailed comparison report including differences between the two models. This can be useful in many scenarios, e.g., say the user wants to compare the changes in model performance after adding a new feature into the model. To use ``rsmcompare``, the user must first run two experiments using either :ref:`rsmtool <usage_rsmtool>` or :ref:`rsmeval <usage_rsmeval>`. ``rsmcompare`` can then be used to compare the outputs of these two experiments to each other.

.. note::

    Currently ``rsmcompare`` takes the outputs of the analyses generated during the original experiments and creates comparison tables. These comparison tables were designed with a specific comparison scenario in mind: comparing a baseline model with a model which includes new feature(s). The tool can certianly be used for other comparison scenarios if the researcher feels that the generated comparison output is appropriate.

``rsmcompare`` can be used to compare:

1. Two ``rsmtool`` experiments, or
2. Two ``rsmeval`` experiments, or
3. An ``rsmtool`` experiment with an ``rsmeval`` experiment (in this case, only the evaluation analyses will be compared).


.. note::

    It is strongly recommend that the original experiments as well as the comparison experiment are all done using the same version of RSMTool.

.. include:: tutorial_rsmcompare.rst.inc

Input
"""""

``rsmcompare`` requires a single argument to run an experiment: the path to :ref:`a configuration file <config_file_rsmcompare>`. It can also take an output directory as an optional second argument. If the latter is not specified, ``rsmcompare`` will use the current directory as the output directory.

Here are all the arguments to the ``rsmcompare`` command-line script.

.. program:: rsmcompare

.. option:: config_file

    The :ref:`JSON configuration file <config_file_rsmcompare>` for this experiment.

.. option:: output_dir (optional)

    The output directory where the report files for this comparison will be stored.

.. option:: -h, --help

    Show help message and exist.

.. option:: -V, --version

    Show version number and exit.

.. include:: config_rsmcompare.rst.inc

Output
""""""
``rsmcompare`` produces the comparison report in HTML format as well as in the form of a Jupyter notebook (a ``.ipynb`` file) in the output directory.
