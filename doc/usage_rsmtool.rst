.. _usage_rsmtool:

Using RSMTool
=============

For most users, the primary means of using RSMTool will be via the command-line utility ``rsmtool``. We refer to each run of ``rsmtool`` as an "experiment".


Input
-----
``rsmtool`` requires a single argument to run an experiment: the path to :ref:`a configuration file <config_file_rsmtool>`. It can also take an output directory as an optional second argument. If the latter is not specified, ``rsmtool`` will use the current directory as the output directory.

Here are all the arguments to the ``rsmtool`` command-line script.

.. program:: rsmtool

.. option:: config_file

    The :ref:`JSON configuration file <config_file_rsmtool>` for this experiment.

.. option:: output_dir (optional)

    The output directory where all the files for this experiment will be stored.

.. option:: -f, --force

    If specified, the contents of the output directory will be overwritten even if it already contains the output of another rsmtool experiment.

.. option:: -h, --help

    Show help message and exist.

.. option:: -V, --version

    Show version number and exit.


.. include:: config_rsmtool.rst.inc

.. _output_dirs_rsmtool:

Output
------

``rsmtool`` produces a set of folders in the experiment output directory. This is either the current directory in which ``rsmtool`` is run or the directory specified as the second optional command-line argument.
If logging to `Weights & Biases <https://wandb.ai/>`__ is :ref:`enabled <use_wandb>`, the reports and all intermediate files are also logged to
the specified Weights & Biases project.

report
^^^^^^
This folder contains the final RSMTool report in HTML format as well in the form of a Jupyter notebook (a ``.ipynb`` file).

output
^^^^^^
This folder contains all of the :ref:`intermediate files <intermediate_files_rsmtool>` produced as part of the various analyses performed, saved as ``.csv`` files. ``rsmtool`` will also save in this folder a copy of the :ref:`configuration file <config_file_rsmtool>`. Fields not specified in the original configuration file will be pre-populated with default values.

figure
^^^^^^
This folder contains all of the figures generated as part of the various analyses performed, saved as ``.svg`` files.

feature
^^^^^^^
This folder contains a ``.csv`` file that lists all features, signs and transformation as used in the `final` model, taking into account any manual or automatic feature selection. See :ref:`feature column selection <column_selection_rsmtool>` for more details.

.. include:: column_selection_rsmtool.rst.inc

.. include:: intermediate_files_rsmtool.rst.inc

.. include:: builtin_models.rst.inc
