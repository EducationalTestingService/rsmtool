.. _config_file_rsmexplain:

Experiment configuration file
"""""""""""""""""""""""""""""

This is a file in ``.json`` format that provides overall configuration options for an ``rsmexplain`` experiment. Here's an `example configuration file <https://github.com/EducationalTestingService/rsmtool/blob/main/examples/rsmexplain/config_rsmexplain.json>`__ for ``rsmexplain``.

.. note:: To make it easy to get started with  ``rsmexplain``, we provide a way to **automatically generate** configurations file both interactively as well as non-interactively. Novice users will find interactive generation more helpful while more advanced users will prefer non-interactive generation. See :ref:`this page <autogen_configuration>` for more details.

Next, we describe all of the ``rsmexplain`` configuration fields in detail. There are four required fields and the rest are all optional. We first describe the required fields and then the optional ones (sorted alphabetically).

background_data
~~~~~~~~~~~~~~~
The path to the background data feature file in one of the :ref:`supported formats <input_file_format>`. Each row should correspond to a single response and contain the numeric feature values extracted for this response. In addition, there should be a column containing a unique identifier (ID) for each response. This path can be absolute or relative to the location of the config file. It must contain at least 300 responses to ensure meaningful explanations.

explain_data
~~~~~~~~~~~~
The path to the file containing the data that we want to explain. The file should be in one of the :ref:`supported formats <input_file_format>`. Each row should correspond to a single response and contain numeric feature values extracted for this response. In addition, there should be a column containing a unique identifier (ID) for each response. The path can be absolute or relative to the location of config file.

experiment_id
~~~~~~~~~~~~~
An identifier for the ``rsmexplain`` experiment. This will be used to name the report. It can be any combination of alphanumeric values, must *not* contain spaces, and must *not* be any longer than 200 characters.

experiment_dir
~~~~~~~~~~~~~~
The directory containing the rsmtool models we want to explain. This directory should contain an ``output`` sub-directory and that sub-directory should contain two files: the ``<experiment_id>.model`` and  ``<experiment_id>_feature.csv``. Note that ``<experiment_id>`` refers to the one defined in this same configuration file. As an example of this directory structure, take a look at the ``existing_experiment`` directory  `here <https://github.com/EducationalTestingService/rsmtool/tree/main/tests/data/experiments/svr-explain>`__

background_kmeans_size *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The size of the k-means sample for background sampling. Defaults to 500. We summarize the dataset specified in `background_data` with this many k-means clusters (each cluster is weighted by the number of data points it represents) and then use the summarized data set for sampling *instead* of the original. The k-means clustering allows us to speed up the explanation process but may sacrifice some accuracy. The default value of 500 has been shown to provide a good balance between speed and accuracy in our experiments. You may use a higher value if you have a very large or very diverse background dataset and you want to ensure that it's accurately summarized.

.. warning:: ``background_kmeans_size`` *must* be smaller than the size of the original background data. If not, you may see errors like this: ``ValueError: n_samples=500 should be >= n_clusters=750.``

custom_sections *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A list of custom, user-defined sections to be included into the final report. These are IPython notebooks (``.ipynb`` files) created by the user.  The list must contains paths to the notebook files, either absolute or relative to the configuration file. All custom notebooks have access to some :ref:`pre-defined variables <custom_notebooks>`.

description *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~
A brief description of the ``rsmexplain`` experiment that will be shown at the top of the report. The description can contain spaces and punctuation.

num_features_to_display *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Number of top features that should be displayed in ``rsmexplain`` plots. Defaults to 15.

.. _rsmexplain_id_column:

id_column *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~
The name of the column containing the response IDs. Defaults to ``spkitemid``, i.e., if this is not specified, ``rsmexplain`` will look for a column called ``spkitemid`` in both ``background_data`` and ``explain_data`` files. Note that the name of the ``id_column`` must be the same in these two files.

.. _rsmexplain_sample_range:

sample_range *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~
If we want to explain a specific range of responses from the ``explain_data``, we can specify that range here. Note that the range is specified in terms of the *location* of the responses in the ``explain_data`` file and that the locations are zero-indexed. So, for example, to explain only the first 50 responses in the file, we should set a value of ``"0-49"`` for this option. Defaults to ``None``.

.. _rsmexplain_sample_size:

sample_size *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~
If we want to explain a random sample of the responses in explain_data, we can specify the size of that random sample here. For example, to explain a random sample of 10 responses, we would set this to 10. Defaults to ``None``.

.. _rsmexplain_sample_ids:

sample_ids *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~
If we want to explain a specific set of responses from the ``explain_data``, we can specify their IDs here as a comma-separated string. Note that the IDs must be values from the :ref:`id_column <rsmexplain_id_column>`. For example, if ``explain_data`` has IDs of the form ``"EXAMPLE_1"``, ``"EXAMPLE_2"`` etc. and we want to explain the fifth, tenth, and twelfth example, the value of this field must be set to ``"EXAMPLE_5, EXAMPLE_10, EXAMPLE_12"``. Defaults to ``None``.

.. note ::

    Only one of ``sample_ids``, ``sample_range`` or ``sample_size`` must be specified. If none of them are specified, explanations will be generated for the entire set of responses in ``explain_data`` which could be very slow, depending on its size.

show_auto_cohorts *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If this option is set to ``true``, auto cohort bar plots will be displayed. These plots can be useful for detecting interaction effects between cohorts and features. If a cohort shows a high feature value, then there may be an interaction between that cohort and the feature. Defaults to ``false``. These plots are not shown by default because these plots may be unstable or provide misleading information if ``explain_data`` is not large enough. For smaller datasets, SHAP may not be able to detect strong feature interactions and compute clear cohorts. If that happens, the plots wil be too specific to be useful.

standardize_features *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If this option is set to ``false``, the feature values for the responses in ``background_data`` and ``explain_data`` will *not* be standardized using the mean and standard deviation parameters for the rsmtool experiment. These parameters are expected to be part of the feature information contained in ``<experiment_dir>/output/<experiment_id>_feature.csv``. Defaults to ``true``.

.. important::

    If ``experiment_dir`` contains the rsmtool configuration file, that file's value for ``standardize_features`` will override the value specified by the user. The reason is that if ``rsmtool`` trained the model with (or without) standardized features, then ``rsmexplain`` must do the same for the explanations to be meaningful.

.. _use_wandb_rsmexplain:

use_wandb *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~

If set to ``true``, the generated reports and all intermediate tables will be logged to `Weights & Biases <https://wandb.ai/>`__.
The Weights & Biases :ref:`entity <wandb_entity>` and :ref:`project name <wandb_project>` should be specified in the appropriate configuration fields. Defaults to ``false``.

.. _wandb_project_rsmexplain:

wandb_project *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~
The Weights & Biases project name if logging to Weights & Biases is enabled. If a project by this name does not already exist, it will be created.

.. important::
    1. Before using Weights & Biases for the first time, users should log in and provide their API key as described in
       `W&B Quickstart guidelines <https://docs.wandb.ai/quickstart#2-log-in-to-wb>`__.
    2. Note that when using W&B logging, the rsmtool run may take significantly longer due to the network traffic being
       sent to W&B.

.. _wandb_entity_rsmexplain:

wandb_entity *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~
The Weights & Biases entity name if logging to Weights & Biases is enabled.
Entity can be a user name or the name of a team or organization.
