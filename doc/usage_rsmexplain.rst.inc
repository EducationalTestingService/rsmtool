.. _usage_rsmexplain:

``rsmexplain`` - Explain non-linear models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RSMTool provides the ``rsmexplain`` command-line utility to generate explanation report for explain data using a model already trained using the ``rsmtool`` utility. This can be useful how the model decision have been impacted by each feature and weight of each feature in the final target.

``rsmexplain`` The sampling explainer is model agnostic and should, in principle, work for any type of model. RSMExplain currently only supports regressors.

.. note::
    No score is generated for responses with non-numeric values for any of the features included into the model.

    If the original model specified transformations for some of the features and these transformations led to ``NaN`` or ``Inf`` values when applied to the new data, ``rsmexplain`` will raise a warning.


.. include:: tutorial_rsmexplain.rst.inc

Input
"""""
``rsmexplain`` requires one arguments to generate predictions: the path to :ref:`a configuration file <config_file_rsmexplain>`.

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

``rsmexplain`` produces a set of folders in the output directory.

report
~~~~~~
This folder contains the final RSMExplain report in HTML format as well as in the form of a Jupyter notebook (a ``.ipynb`` file).

output
~~~~~~
This folder contains various shap values and absolute value variants of the SHAP values, saved as ``.csv`` files. ``rsmexplain`` will also save in this folder a copy of the :ref:`configuration file <config_file_rsmexplain>`. Fields not specified in the original configuration file will be pre-populated with default values. The shap explanation object is saved as ``<Experiment_id>_explanation.pkl`` and a mapping between the position of the example in the dataset and the ID of the example is saved as ``<Experiment_id>_ids.pkl``

figure
~~~~~~
This folder contains all of the figures generated as part of the various shap plots, saved as ``.svg`` files.
