.. _usage_rsmeval:

``rsmeval`` - Evaluate external predictions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RSMTool provides the ``rsmeval`` command-line utility to evaluate existing predictions and generate a report with all the built-in analyses. This can be useful in scenarios where the user wants to use more sophisticated machine learning algorithms not available in RSMTool to build the scoring model but still wants to be able to evaluate that model's predictions using the standard analyses.

For example, say a researcher *has* an existing automated scoring engine for grading short responses that extracts the features and computes the predicted score. This engine uses a large number of binary, sparse features. She cannot use ``rsmtool`` to train her model since it requires numeric features. So, she uses `scikit-learn <https://scikit-learn.org/>`_ to train her model.

Once the model is trained, the researcher wants to evaluate her engine's performance using the analyses recommended by the educational measurement community as well as conduct additional investigations for specific subgroups of test-takers. However, these kinds of analyses are not available in ``scikit-learn``. She can use ``rsmeval`` to set up a customized report using a combination of existing and custom sections and quickly produce the evaluation that is useful to her.

.. include:: tutorial_rsmeval.rst.inc

Input
"""""

``rsmeval`` requires a single argument to run an experiment: the path to :ref:`a configuration file <config_file_rsmeval>`. It can also take an output directory as an optional second argument. If the latter is not specified, ``rsmeval`` will use the current directory as the output directory.

Here are all the arguments to the ``rsmeval`` command-line script.

.. program:: rsmeval

.. option:: config_file

    The :ref:`JSON configuration file <config_file_rsmeval>` for this experiment.

.. option:: output_dir (optional)

    The output directory where all the files for this experiment will be stored.

.. option:: -f, --force

    If specified, the contents of the output directory will be overwritten even if it already contains the output of another rsmeval experiment.

.. option:: -h, --help

    Show help message and exist.

.. option:: -V, --version

    Show version number and exit.

.. include:: config_rsmeval.rst.inc

.. _output_dirs_rsmeval:

Output
""""""

``rsmeval`` produces a set of folders in the output directory.

report
~~~~~~
This folder contains the final RSMEval report in HTML format as well as in the form of a Jupyter notebook (a ``.ipynb`` file).

output
~~~~~~
This folder contains all of the :ref:`intermediate files <intermediate_files_rsmeval>` produced as part of the various analyses performed, saved as ``.csv`` files. ``rsmeval`` will also save in this folder a copy of the :ref:`configuration file <config_file_rsmeval>`. Fields not specified in the original configuration file will be pre-populated with default values. 

figure
~~~~~~
This folder contains all of the figures generated as part of the various analyses performed, saved as ``.svg`` files.

.. include:: intermediate_files_rsmeval.rst.inc

