.. _usage_rsmexplain:

``rsmexplain`` - Explain non-linear models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RSMTool provides the ``rsmexplain`` command-line utility to generate a report explaining the predictions made by a model trained using the ``rsmtool`` utility. These explanations contain useful information about the contribution of each feature to the final score, even if the model is non-linear or black-box in nature. The `rsmexplain` command-line utility uses the `SHAP library <https://shap.readthedocs.io/en/latest/index.html>`_ to compute the explanations.

.. note::

    ``rsmexplain`` uses the sampling explainer which is model agnostic and should, in principle, work for any type of model. However, ``rsmexplain`` currently only supports regressors since they are the most popular model type used for automated scoring.

.. include:: tutorial_rsmexplain.rst.inc

Input
"""""
``rsmexplain`` requires only one argument to generate the explanation report: the path to :ref:`a configuration file <config_file_rsmexplain>`.

Here are all the arguments to the ``rsmexplain`` command-line script.

.. program:: rsmpredict

.. option:: config_file

    The :ref:`JSON configuration file <config_file_rsmexplain>` for this experiment.

.. option:: output_dir

    The output directory where all the files for this experiment will be stored.

.. option:: -f, --force

    If specified, the contents of the output directory will be overwritten even if it already contains the output of another rsmeval experiment.

.. option:: -h, --help

    Show help message and exist.

.. option:: -V, --version

    Show version number and exit.

.. include:: config_rsmexplain.rst.inc

.. _output_dirs_rsmexplain:

Output
""""""

``rsmexplain`` produces a set of folders in the output directory. If logging to `Weights & Biases <https://wandb.ai/>`__ is :ref:`enabled <use_wandb_rsmexplain>`,
the reports are also logged to the specified Weights & Biases project.

report
~~~~~~
This folder contains the final explanation report in HTML format as well as in the form of a Jupyter notebook (a ``.ipynb`` file).

output
~~~~~~
This folder contains various SHAP values and their absolute value variants. ``rsmexplain`` also saves  a copy of the :ref:`configuration file <config_file_rsmexplain>` in this folder. Fields not specified in the original configuration file will be pre-populated with default values. The SHAP explanation object is saved as ``<experiment_id>_explanation.pkl`` and a mapping between the position of each explained response in the data file and its unique ID is saved in ``<experiment_id>_ids.pkl``.

figure
~~~~~~
This folder contains all of the figures containing the various SHAP plots, saved as ``.svg`` files.
