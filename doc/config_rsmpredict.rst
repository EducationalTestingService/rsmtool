Experiment configuration file
"""""""""""""""""""""""""""""

This is a file in ``.json`` format that provides overall configuration options for an ``rsmpredict`` experiment. An example configuration file can be found here.

There are three required fields and the rest are all optional.

experiment_id
~~~~~~~~~~~~~
The original ``experiment_id`` used to create the model files that you will be using for generating predictions. If you do not know the ``experiment_id``, you can find it by looking at the shared first part of the model file names. 

experiment_dir
~~~~~~~~~~~~~~~
The path to the directory which contains the outputs of the original `rsmtool` experiment. This directory must contain a directory called ``output`` with the model files, feature pre-processing parameters and score pre-processing parameters. The path can be absolute or relative to the location of config file.

input_feature_file
~~~~~~~~~~~~~~~~
the path to the file with raw feature values that will be used to generate predictions in .csv format. Each row should correspond to a single response and contain feature values for this response. In addition, there should be a column with a unique identifier (ID) for each response. The path can be absolute or relative to the location of config file. Note that the feature names must be the same as used in the original ``rsmtool`` experiment. 


id_column *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~

The name of the column containing the response IDs. Defaults to ``spkitemid``, i.e., if this is not specified, ``rsmpredict`` will look for a column called ``spkitemid`` in the prediction file.


Columns that can be added to the predictions (optional)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following columns are not used by ``rsmpredict``. You can specify in the config file so that they are included into the predictions file for subsequent use with ``rsmeval``. Different fields are used for different types of columns for consistency with other tools.

    - ``candidate_column (optional)`` - the name for the column containing unique candidate IDs. This column will be named ``candidate`` in the output file with predictions.

    - ``human_score_column (optional)`` - the name for the column containing human scores. This column will be renamed to ``sc1``. 

    - ``second_human_score_column (optional)`` - the name for the column containing the second human score. This column will be renamed to ``sc2``. 

    - ``subgroups (optional)`` -  a list of grouping variables for generating future analyses by prompt or subgroup analyses. For example, ``["prompt, gender, native_language, test_country"]``. All these columns will be included into the output file with the original names.

    - ``flag_column (optional)`` - see configuration file for `rsmtool <config_file>` for further information. No filtering will be done by `rsmpredict`, but the content of the flag columns will be added to the predictions file using the original column names for future use with `rsmeval`. 

