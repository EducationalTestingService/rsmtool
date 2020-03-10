.. _config_file_rsmpredict:

Experiment configuration file
"""""""""""""""""""""""""""""
This is a file in ``.json`` format that provides overall configuration options for an ``rsmpredict`` experiment. Here's an `example configuration file <https://github.com/EducationalTestingService/rsmtool/blob/master/examples/rsmpredict/config_rsmpredict.json>`_ for ``rsmpredict``. To make it easy to get started with  ``rsmpredict``, we provide a way to automatically generate a configuration file that you can then just edit based on your data and your needs. To do so, simply run ``rsmpredict generate`` at the commmand line. Next, we describe all of the ``rsmpredict`` configuration fields in detail.

There are three required fields and the rest are all optional. We first describe the required fields and then the optional ones (sorted alphabetically).

experiment_dir
~~~~~~~~~~~~~~
The path to the directory containing ``rsmtool`` model to use for generating predictions. This directory must contain a sub-directory called ``output`` with the model files, feature pre-processing parameters, and score post-processing parameters. The path can be absolute or relative to the location of configuration file.

experiment_id
~~~~~~~~~~~~~
The ``experiment_id`` used to create the ``rsmtool`` model files being used for generating predictions. If you do not know the ``experiment_id``, you can find it by looking at the prefix of the ``.model`` file under the ``output`` directory.

input_feature_file
~~~~~~~~~~~~~~~~~~
The path to the file with the raw feature values that will be used for generating predictions. The file should be in one of the :ref:`supported formats <input_file_format>` Each row should correspond to a single response and contain feature values for this response. In addition, there should be a column with a unique identifier (ID) for each response. The path can be absolute or relative to the location of config file. Note that the feature names *must* be the same as used in the original ``rsmtool`` experiment.

.. note::

    ``rsmpredict`` will only generate predictions for responses in this file that have numeric values for the features included in the ``rsmtool`` model.

.. seealso::

    ``rsmpredict`` does not require human scores for the new data since it does not evaluate the generated predictions. If you do have the human scores and want to evaluate the new predictions, you can use the :ref:`rsmeval <usage_rsmeval>` command-line utility.

candidate_column *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The name for the column containing unique candidate IDs. This column will be named ``candidate`` in the output file with predictions.

.. _file_format_predict:

file_format *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~
The format of the :ref:`intermediate files <intermediate_files_rsmtool>`. Options are ``csv``, ``tsv``, or ``xlsx``. Defaults to ``csv`` if this is not specified.

flag_column *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~
See description in the :ref:`rsmtool configuration file <flag_column_rsmtool>` for further information. No filtering will be done by ``rsmpredict``, but the contents of all specified columns will be added to the predictions file using the original column names.

human_score_column *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The name for the column containing human scores. This column will be renamed to ``sc1``.

id_column *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~
The name of the column containing the response IDs. Defaults to ``spkitemid``, i.e., if this is not specified, ``rsmpredict`` will look for a column called ``spkitemid`` in the prediction file.

There are several other options in the configuration file that, while not directly used by ``rsmpredict``, can simply be passed through from the input features file to the output predictions file. This can be particularly useful if you want to subsequently run :ref:`rsmeval <usage_rsmeval>` to evaluate the generated predictions.

predict_expected_scores *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If the original model was a probabilistic SKLL classifier, then *expected scores* --- probability-weighted averages over a contiguous, numeric score points --- can be generated as the machine predictions instead of the most likely score point, which would be the default. Set this field to ``true`` to compute expected scores as predictions. Defaults to ``false``.

.. note::

    1. If the model in the original ``rsmtool`` experiment is an SVC, that original experiment *must* have been run with ``predict_expected_scores`` set to ``true``. This is because SVC classifiers are fit differently if probabilistic output is desired, in contrast to other probabilistic SKLL classifiers.

    2. You may see slight differences in expected score predictions if you run the experiment on different machines or on different operating systems most likely due to very small probability values for certain score points which can affect floating point computations.

second_human_score_column *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The name for the column containing the second human score. This column will be renamed to ``sc2``.

standardize_features *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If this option is set to ``false`` features will not be standardized by dividing by the mean and multiplying by the standard deviation. Defaults to ``true``.

subgroups *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~
A list of column names indicating grouping variables used for generating analyses specific to each of those defined subgroups. For example, ``["prompt, gender, native_language, test_country"]``. All these columns will be included into the predictions file with the original names.
