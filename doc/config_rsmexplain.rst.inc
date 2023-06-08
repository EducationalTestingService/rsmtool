.. _config_file_rsmexplain:

Experiment configuration file
"""""""""""""""""""""""""""""

This is a file in ``.json`` format that provides overall configuration options for an ``rsmexplain`` experiment. Here's an `example configuration file <https://github.com/EducationalTestingService/rsmtool/blob/main/examples/rsmexplain/config_rsmexplain.json>`__ for ``rsmexplain``.

.. note:: To make it easy to get started with  ``rsmexplain``, we provide a way to **automatically generate** configurations file both interactively as well as non-interactively. Novice users will find interactive generation more helpful while more advanced users will prefer non-interactive generation. See :ref:`this page <autogen_configuration>` for more details.

Next, we describe all of the ``rsmexplain`` configuration fields in detail. There are four required fields and the rest are all optional. We first describe the required fields and then the optional ones (sorted alphabetically).

background_data
~~~~~~~~~~~~~~~
The path to the background data feature file one of the :ref:`supported formats <input_file_format>`. Each row should correspond to a single response and contain the numeric feature values extracted for this response. In addition, there should be a column containing a unique identifier (ID) for each response. This path can be absolute or relative to the location of config file. It must contain at least 300 responses to ensure meaningful explanations.

explainable_data
~~~~~~~~~~~~~~~~
The path to the file containing the data that we want to explain. The file should be in one of the :ref:`supported formats <input_file_format>`. Each row should correspond to a single response and contain numeric feature values extracted for this response. In addition, there should be a column containing a unique identifier (ID) for each response. The path can be absolute or relative to the location of config file.

experiment_id
~~~~~~~~~~~~~
An identifier for the ``rsmexplain`` experiment. This will be used to name the report. It can be any combination of alphanumeric values, must *not* contain spaces, and must *not* be any longer than 200 characters.

experiment_dir
~~~~~~~~~~~~~~
The directory containing the rsmtool models we want to explain. This directory should contain an ``output`` sub-directory and that sub-directory should contain two files: the ``<experiment_id>.model`` and  ``<experiment_id>_feature.csv``. Note that ``<experiment_id>`` refers to the one defined in this same configuration file. As an example of this directory structure, take a look at the ``existing_experiment`` directory  `here <https://github.com/EducationalTestingService/rsmtool/tree/main/tests/data/experiments/svr-explain>`__

background_kmeans_size *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The size of k-means sample for background. Defaults to ``500``.

custom_sections *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A list of custom, user-defined sections to be included into the final report. These are IPython notebooks (``.ipynb`` files) created by the user.  The list must contains paths to the notebook files, either absolute or relative to the configuration file. All custom notebooks have access to some :ref:`pre-defined variables <custom_notebooks>`.

description *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~
A brief description of the explanation. The description can contain spaces and punctuation.

num_features_to_display *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Number of top features should be displayed in rsmexplain plots. Defaults to ``15``.

id_column *(Optional)*
~~~~~~~~~
The name of the column containing the response IDs. Defaults to ``spkitemid``, i.e., if this is not specified, ``rsmexplain`` will look for a column called ``spkitemid`` in both background_data and explainable_data files. Note: id_column must be same in these two files.

sample_range *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~
Range of specific sample IDs to be explained. i.e., ``"1-50"`` indicates instances from the first to the 50th. Defaults to ``None``.

sample_size *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~
Size of random sample to be explained. i.e., ``10`` indicates to randomly sample 10 instances from the explainable data. Defaults to ``None``.

.. note ::

    If neither ``sample_range`` nor ``sample_size`` is specified, explanations will be generated for the entire explainable_data which could be very slow, depending on its size.

show_auto_cohorts *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If this option is set to ``True``, auto cohort bar plots will be displayed. These plots can be useful to detect interaction effects between cohorts and features. If a cohort shows a high feature value, then there may be an interaction between that cohort and the feature. Defaults to ``False``.

standardize_features *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If this option is set to ``false`` features will not be standardized by dividing by the mean and multiplying by the standard deviation. Defaults to ``true``.
