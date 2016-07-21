RSMEval
^^^^^^^

`rsmeval` evaluates the existing predictions and creates a report with all standard evaluations. Here is the most common scenario:

A researcher *has* an automated scoring system for grading short responses that extracts the features and computes the score. He wants to evaluate the system performance using metrics commonly used in educational community but not always available in standard machine learning packages as well as to conduct additional analyses to evaluate system fairness and compare system performance to human-human agreement.
He can then use RSMEval to set up customized evaluation report using a combination of existing and custom sections and quickly produce a new report for each version of his system. 

Input
"""""

``rsmeval`` requires a single argument to run an experiment: the path to the configuration file. It can also take an output directory as an optional second argument. If the latter is not specified, ``rsmeval`` will use the current directory as the output directory.

.. include:: config_rsmeval.rst

Output
""""""

``rsmeval`` produces set of folders in the output directory.

report
~~~~~~
This folder contains the final RSMEval report in HTML format as well as a Jupyter notebook (a ``.ipynb`` file).

output
~~~~~~
This folder contains all of the :ref:`intermediate files <intermediate_files_rsmeval>` produced as part of the various analyses performed, saved as ``.csv`` files.


figure
~~~~~~
This folder contains all of the figures generated as part of the various analyses performed, saved as ``.svg`` files.


.. include:: intermediate_files_rsmeval.rst
