.. _config_file_rsmpredict:

Experiment configuration file
"""""""""""""""""""""""""""""
This is a file in ``.json`` format that provides overall configuration options for an ``rsmpredict`` experiment. Here's an example configuration file for `rsmpredict <https://github.com/EducationalTestingService/rsmtool/blob/master/examples/rsmpredict/config_rsmpredict.json>`_.

There are three required fields and the rest are all optional.

experiment_dir
~~~~~~~~~~~~~~
The path to the directory containing ``rsmtool`` model to use for generating predictions. This directory must contain a sub-directory called ``output`` with the model files, feature pre-processing parameters, and score post-processing parameters. The path can be absolute or relative to the location of configuration file.

experiment_id
~~~~~~~~~~~~~
The ``experiment_id`` used to create the ``rsmtool`` model files being used for generating predictions. If you do not know the ``experiment_id``, you can find it by looking at the prefix of the ``.model`` file under the ``output`` directory.

input_feature_file
~~~~~~~~~~~~~~~~~~
The path to the ``.csv`` file with the raw feature values that will be used for generating predictions. Each row should correspond to a single response and contain feature values for this response. In addition, there should be a column with a unique identifier (ID) for each response. The path can be absolute or relative to the location of config file. Note that the feature names *must* be the same as used in the original ``rsmtool`` experiment.


.. note::

    ``rsmpredict`` will only generate predictions for responses in this file that have numeric values for the features included in the ``rsmtool`` model.


.. seealso::

    ``rsmpredict`` does not require human scores for the new data since it does not evaluate the generated predictions. If you do have the human scores and want to evaluate the new predictions, you can use the :ref:`rsmeval <usage_rsmeval>` command-line utility.


id_column *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~

The name of the column containing the response IDs. Defaults to ``spkitemid``, i.e., if this is not specified, ``rsmpredict`` will look for a column called ``spkitemid`` in the prediction file.

There are several other options in the configuration file that, while not directly used by ``rsmpredict``, can simply be passed through from the input features file to the output predictions file. This can be particularly useful if you want to subsequently run :ref:`rsmeval <usage_rsmeval>` to evaluate the generated predictions.

candidate_column *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The name for the column containing unique candidate IDs. This column will be named ``candidate`` in the output file with predictions.

human_score_column *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The name for the column containing human scores. This column will be renamed to ``sc1``.

second_human_score_column *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The name for the column containing the second human score. This column will be renamed to ``sc2``.

subgroups *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~
A list of column names indicating grouping variables used for generating analyses specific to each of those defined subgroups. For example, ``["prompt, gender, native_language, test_country"]``. All these columns will be included into the predictions file with the original names.

flag_column *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~
See description in the :ref:`rsmtool configuration file <flag_column_rsmtool>` for further information. No filtering will be done by ``rsmpredict``, but the contents of all specified columns will be added to the predictions file using the original column names.

