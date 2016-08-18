.. _usage_rsmsumm:

``rsmsumm`` - Compare multiple scoring models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RSMTool provides the ``rsmsumm`` command-line utility to compare multiple models and to generate a comparison report. Unlike ``rsmcompare`` which creates a detailed comparison between the two models, ``rsmsumm`` can be used to create a more general overview of multiple models. 

``rsmsumm`` can be used to compare:

1. Multiple ``rsmtool`` experiments, or
2. Multiple ``rsmeval`` experiments, or
3. A mix of ``rsmtool`` and ``rsmeval`` experiments (in this case, only the evaluation analyses will be compared).


.. note::

    It is strongly recommend that the original experiments as well as the summary experiment are all done using the same version of RSMTool.

.. include:: tutorial_rsmsumm.rst

Input
"""""

When used without any arguments, ``rsmsumm`` will compare all models in the current directory and generate a comparison report named ``model_comparison_report.html``. You can specify which models you want to compare and the name of the report by supplying the path to :ref:`a configuration file <config_file_rsmsumm>`. It can also take an output directory as an optional second argument. If the latter is not specified, ``rsmsumm`` will use the current directory as the output directory.

Here are all the arguments to the ``rsmsumm`` command-line script.

.. program:: rsmsumm

.. option:: config_file (optional)

    The :ref:`JSON configuration file <config_file_rsmsumm>` for this experiment.

.. option:: output_dir (optional)

    The output directory where the report files for this comparison will be stored.

.. option:: -h, --help

    Show help message and exist.

.. option:: -V, --version

    Show version number and exit.

.. include:: config_rsmcompare.rst

Output
""""""
``rsmsumm`` produces the comparison report in HTML format as well as in the form of a Jupyter notebook (a ``.ipynb`` file) in the output directory.
