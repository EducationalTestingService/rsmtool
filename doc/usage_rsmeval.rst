.. _usage_rsmeval:

``rsmeval`` - Evaluate external predictions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RSMTool provides the ``rsmeval`` command-line utility to evaluate existing predictions and generate a report with all the built-in analyses. This can be useful in scenarios where the user wants to use more sophisticated machine learning algorithms not available in RSMTool to build the scoring model but still wants to be able to evaluate that model's predictions using the standard analyses.

For example, say a researcher *has* an existing automated scoring engine for grading short responses that extracts the features and computes the predicted score. This engine uses a large number of binary, sparse features. She cannot use ``rsmtool`` to train her model since it requires numeric features. So, she uses `scikit-learn <http://scikit-learn.org/>`_ to train her model.

Once the model is trained, the researcher wants to evaluate her engine's performance using the analyses recommended by the educational measurement community as well as conduct additional investigations for specific subgroups of test-takers. However, these kinds of analyses are not available in ``scikit-learn``. She can use ``rsmeval`` to set up a customized report using a combination of existing and custom sections and quickly produce the evaluation that is useful to her.

Input
"""""

``rsmeval`` requires a single argument to run an experiment: the path to a configuration file. It can also take an output directory as an optional second argument. If the latter is not specified, ``rsmeval`` will use the current directory as the output directory.

.. include:: config_rsmeval.rst

Output
""""""

``rsmeval`` produces set of folders in the output directory.

report
~~~~~~
This folder contains the final RSMEval report in HTML format as well as in the form of a Jupyter notebook (a ``.ipynb`` file).

output
~~~~~~
This folder contains all of the :ref:`intermediate files <intermediate_files_rsmeval>` produced as part of the various analyses performed, saved as ``.csv`` files.

figure
~~~~~~
This folder contains all of the figures generated as part of the various analyses performed, saved as ``.svg`` files.

.. include:: intermediate_files_rsmeval.rst
