Using RSMTool
=============

For most users, the primary means of using RSMTool will be via the command-line utility ``rsmtool``. We refer to each run of ``rsmtool`` as an "experiment".

Input
-----

``rsmtool`` requires a single argument to run an experiment: the path to the configuration file. It can also take an output directory as an optional second argument. If the latter is not specified, ``rsmtool`` will use the current directory as the output directory.

.. include:: config_rsmtool.rst

Output
------

``rsmtool`` produces set of folders in the experiment output directory. This is either the current directory in which ``rsmtool`` is run or the directory specified as the second optional command-line argument.

report
^^^^^^
This folder contains the final RSMTool report in HTML format as well in the form of a Jupyter notebook (a ``.ipynb`` file).

output
^^^^^^
This folder contains all of the :ref:`intermediate files <intermediate_files_rsmtool>` produced as part of the various analyses performed, saved as ``.csv`` files.

figure
^^^^^^
This folder contains all of the figures generated as part of the various analyses performed, saved as ``.svg`` files.

feature *(Optional)*
^^^^^^^^^^^^^^^^^^^^
This folder is only created if you use :ref:`feature column selection <column_selection_rsmtool>` or a :ref:`model with automatic feature selection <automatic_feature_selection_models>`.

.. include:: column_selection_rsmtool.rst

.. include:: intermediate_files_rsmtool.rst

.. include:: builtin_models.rst
