.. _config_file_rsmtool:

Experiment configuration file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a file in ``.json`` format that provides overall configuration options for an ``rsmtool`` experiment. Here's an example configuration file for `rsmtool <https://github.com/EducationalTestingService/rsmtool/blob/master/examples/rsmtool/config_rsmtool.json>`_.

There are four required fields and the rest are all optional.

experiment_id
"""""""""""""
An identifier for the experiment that will be used to name the report and all :ref:`intermediate files <intermediate_files_rsmtool>`. It can be any combination of alphanumeric values, must *not* contain spaces, and must *not* be any longer than 200 characters.

model
"""""
The machine learner you want to use to build the scoring model. Possible values include :ref:`built-in linear regression models <builtin_models>` as well as all of the learners available via `SKLL <https://skll.readthedocs.io/en/latest/run_experiment.html#learners>`_. With SKLL learners, you can customize the :ref:`tuning objective <skll_objective>` and also :ref:`compute expected scores as predictions <predict_expected_scores>`.

train_file
""""""""""
The path to the training data feature file in one of the :ref:`supported formats <input_file_format>`. Each row should correspond to a single response and contain numeric feature values extracted for this response. In addition, there should be a column with a unique identifier (ID) for each response and a column with the human score for each response. The path can be absolute or relative to the location of config file.

test_file
"""""""""
The path to the evaluation data feature file in one of the :ref:`supported formats <input_file_format>`. Each row should correspond to a single response and contain numeric feature values extracted for this response. In addition, there should be a column with a unique identifier (ID) for each response and a column with the human score for each response. The path can be absolute or relative to the location of config file.

.. _skll_objective:

skll_objective *(Optional)*
"""""""""""""""""""""""""""
The tuning objective to use if a SKLL model is chosen to build the scoring model. Possible values are the objectives available via `SKLL <https://skll.readthedocs.io/en/latest/run_experiment.html#objectives>`_. Defaults to ``neg_mean_squared_error`` for SKLL regressors and ``f1_score_micro`` for SKLL classifiers. Note that if this option is specified with the :ref:`built-in linear regression models <builtin_models>`, it will simply be ignored. 

.. _predict_expected_scores:

predict_expected_scores *(Optional)*
""""""""""""""""""""""""""""""""""""
If a probabilistic SKLL classifier is chosen to build the scoring model, then *expected scores* --- probability-weighted averages over contiguous, numeric score points --- can be generated as the machine predictions instead of the most likely score point, which would be the default for a classifier. Set this field to ``true`` to compute expected scores as predictions. Defaults to ``false``.

.. note ::

    You may see slight differences in expected score predictions if you run the experiment on different machines or on different operating systems most likely due to very small probablity values for certain score points which can affect floating point computations.


description *(Optional)*
""""""""""""""""""""""""
A brief description of the experiment. This will be included in the report. The description can contain spaces and punctuation. It's blank by default.


.. _file_format:

file_format *(Optional)*
"""""""""""""""""""""""""""
The format of the :ref:`intermediate files <intermediate_files_rsmtool>`. Options are ``csv``, ``tsv``, or ``xlsx``. Defaults to ``csv`` if this is not specified.

.. _feature_fields_note:

.. note ::

    By default, ``rsmtool`` will use all of the columns present in the training and evaluation files as features except for any columns explicitly identified in the configuration file (see below). The following four fields (``features``, ``feature_subset_file``, ``feature_subset``, and ``sign``) are useful if you want to use only a specific set of columns as features. See :ref:`selecting feature columns <column_selection_rsmtool>` for more details.


.. _feature_file_rsmtool:

features *(Optional)*
"""""""""""""""""""""
Path to the file with list of features if using :ref:`fine-grained column selection <feature_list_column_selection>`. Alternatively, you can pass a ``list`` of feature names to include in the experiment.

.. _feature_subset_file:

feature_subset_file *(Optional)*
""""""""""""""""""""""""""""""""
Path to the feature subset file if using :ref:`subset-based column selection <subset_column_selection>`.

.. _feature_subset:

feature_subset *(Optional)*
"""""""""""""""""""""""""""
Name of the pre-defined feature subset to be used if using :ref:`subset-based column selection <subset_column_selection>`.

.. _sign:

sign *(Optional)*
"""""""""""""""""
Name of the column containing expected correlation sign between each feature and human score if using :ref:`subset-based column selection <subset_column_selection>`.


.. _id_column_rsmtool:

id_column *(Optional)*
""""""""""""""""""""""
The name of the column containing the response IDs. Defaults to ``spkitemid``, i.e., if this is not specified, ``rsmtool`` will look for a column called ``spkitemid`` in the training and evaluation files.

.. _train_label_column_rsmtool:

train_label_column *(Optional)*
"""""""""""""""""""""""""""""""
The name for the column containing the human scores in the training data. If set to to ``fake``, fake scores will be generated using randomly sampled integers. This option may be useful if you only need descriptive statistics for the data and do not care about the other analyses. Defaults to ``sc1``.

.. _test_label_column_rsmtool:

test_label_column *(Optional)*
""""""""""""""""""""""""""""""
The name for the column containing the human scores in the training data. If set to to ``fake``, fake scores will be generated using randomly sampled integers. This option may be useful if you only need descriptive statistics for the data and do not care about the other analyses. Defaults to ``sc1``.

.. note::

    All responses with non-numeric values in either ``train_label_column`` or ``test_label_column`` and/or those with non-numeric values for relevant features will be automatically excluded from model training and evaluation. By default, zero scores in either ``train_label_column`` or ``test_label_column`` will also be excluded. See :ref:`exclude_zero_scores_rsmtool` if you want to keep responses with zero scores.

.. _length_column_rsmtool:

length_column *(Optional)*
""""""""""""""""""""""""""
The name for the optional column in the training and evaluation data containing response length. If specified, length is included in the inter-feature and partial correlation analyses. Note that this field *should not* be specified if you want to use the length column as an actual feature in the model. In the latter scenario, the length column will automatically be included in the analyses, like any other feature. If you specify ``length_column`` *and* include the same column name as  a feature in the :ref:`feature file <feature_file_rsmtool>`, ``rsmtool`` will ignore the ``length_column`` setting. In addition, if ``length_column`` has missing values or if its standard deviation is 0 (both somewhat unlikely scenarios), ``rsmtool`` will *not* include any length-based analyses in the report.


.. _second_human_score_column_rsmtool:

second_human_score_column *(Optional)*
""""""""""""""""""""""""""""""""""""""
The name for an optional column in the test data containing a second human score for each response. If specified, additional information about human-human agreement and degradation will be computed and included in the report. Note that this column must contain either numbers or be empty. Non-numeric values are *not* accepted. Note also that the :ref:`exclude_zero_scores_rsmtool` option below will apply to this column too.

.. note::

    You do not need to have second human scores for *all* responses to use this option. The human-human agreement statistics will be computed as long as there is at least one response with numeric value in this column. For responses that do not have a second human score, the value in this column should be blank.

    
.. _flag_column_rsmtool:

flag_column *(Optional)*
""""""""""""""""""""""""
This field makes it possible to only use responses with particular values in a given column (e.g. only responses with a value of ``0`` in a column called ``ADVISORY``). The field takes a dictionary in Python format where the keys are the names of the columns and the values are lists of values for responses that will be used to train the model. For example, a value of ``{"ADVISORY": 0}`` will mean that ``rsmtool`` will *only* use responses for which the ``ADVISORY`` column has the value 0. 
If this field is used without ``flag_column_test``, the conditions will be applied to *both* training and evaluation set and the specified columns must be present in both sets. 
When this field is used in conjunction with ``flag_column_test``, the conditions will be applied to *training set only* and the specified columns must be present in the training set.
Defaults to ``None``.

.. note::

    If  several conditions are specified (e.g., ``{"ADVISORY": 0, "ERROR": 0}``) only those responses which satisfy *all* the conditions will be selected for further analysis (in this example, these will be the responses where the ``ADVISORY`` column has a value of 0 *and* the ``ERROR`` column has a value of 0).


.. note::

    When reading the values in the supplied dictionary, ``rsmtool`` treats numeric strings, floats and integers as the same value. Thus ``1``, ``1.0``, ``"1"`` and ``"1.0"`` are all treated as the ``1.0``.


.. _flag_column_test_rsmtool:

flag_column_test *(Optional)*
"""""""""""""""""""""""""""""
This field makes it possible to only use a separate Python flag dictionary for the evaluation set. If this field is not passed, and ``flag_column`` is passed, then the same advisories will be used for both training and evaluation sets. 


When this field is used, the specified columns must be present in the evaluation set. 
Defaults to ``None`` or `flag_column``, if ``flag_column`` is present. Use ``flag_column_test`` only if you want filtering of the test set.

.. note::
    
    When used, ``flag_column_test`` field determines *all* filtering conditions for the evaluation set. If it is used in conjunction with ``flag_column`` field, the filtering conditions defined in ``flag_column`` will *only* be applied to the training set. If you want to apply a subset of conditions to both partitions with additional conditions applied to the evaluation set only, you will need to specify the overlapping conditions separately for each partition.   

.. _exclude_zero_scores_rsmtool:

exclude_zero_scores *(Optional)*
""""""""""""""""""""""""""""""""
By default, responses with human scores of 0 will be excluded from both training and evaluation set. Set this field to ``false`` if you want to keep responses with scores of 0. Defaults to ``true``.

.. _trim_min_rsmtool:

trim_min *(Optional)*
"""""""""""""""""""""
The single numeric value for the lowest possible integer score that the machine should predict. This value will be used to compute the floor value for :ref:`trimmed (bound) <score_postprocessing>` machine scores as ``trim_min`` - ``trim_tolerance``. Defaults to the lowest observed human score in the training data or 1 if there are no numeric human scores available.


.. _trim_max_rsmtool:

trim_max *(Optional)*
"""""""""""""""""""""
The single numeric value for the highest possible integer score that the machine should predict. This value will be used to compute the ceiling value for :ref:`trimmed (bound) <score_postprocessing>` machine scores as ``trim_max`` + ``trim_tolerance``. Defaults to the highest observed human score in the training data or 10 if there are no numeric human scores available.

.. _trim_tolerance_rsmtool:

trim_tolerance *(Optional)*
"""""""""""""""""""""""""""

The single numeric value that will be used to pad the trimming range specified in ``trim_min`` and ``trim_max``. This value will be used to compute the ceiling and floor values for :ref:`trimmed (bound) <score_postprocessing>` machine scores as ``trim_max`` + ``trim_tolerance`` for ceiling value and ``trim_min``-``trim_tolerance`` for floor value.
Defaults to 0.49998.

.. note::
    
    For more fine-grained control over the trimming range, you can set ``trim_tolerance`` to `0` and use ``trim_min`` and ``trim_max`` to specify the exact floor and ceiling values.  

.. _select_transformations_rsmtool:

select_transformations *(Optional)*
"""""""""""""""""""""""""""""""""""
If this option is set to ``true`` the system will try apply feature transformations to each of the features and then choose the transformation for each feature that yields the highest correlation with human score. The possible transformations are:

    * ``raw``: no transformation, use original feature value
    * ``org``: same as raw
    * ``inv``: 1/x
    * ``sqrt``: square root
    * ``addOneInv``: 1/(x+1)
    * ``addOneLn``: ln(x+1)

Note that ``inv`` is never used for features with positive values. Defaults to ``false``.

.. seealso::

    It is also possible to manually apply transformations to any feature as part of the :ref:`feature column selection <feature_list_column_selection>` process.


.. _standardize_features:

standardize_features *(Optional)*
"""""""""""""""""""""""""""""""""
If this option is set to ``false`` features will not be standardized by dividing by the mean and multiplying by the standard deviation. Defaults to ``true``.


.. _use_scaled_predictions_rsmtool:

use_scaled_predictions *(Optional)*
"""""""""""""""""""""""""""""""""""
If set to ``true``, certain evaluations (confusion matrices, score distributions, subgroup analyses) will use the scaled machine scores. If set to ``false``, these evaluations will use the raw machine scores. Defaults to ``false``.

.. note::

    All evaluation metrics (e.g., kappa and pearson correlation) are automatically computed for *both* scaled and raw scores.


.. _use_truncation_thresholds:

use_truncation_thresholds *(Optional)*
""""""""""""""""""""""""""""""""""""""
If set to ``true``, use the ``min`` and ``max`` columns specified in the ``features`` file to clamp outlier feature values. This is useful if users would like to clamp feature values based on some pre-defined boundaries, rather than having these boundaries calculated based on the training set. Defaults to ``false``.

.. note::

    If ``_use_truncation_thresholds`` is set, a ``features`` file _must_ be specified, and this file _must_ include ``min`` and ``max`` columns. If no ``feature`` file is specified or these columns are missing, an error will be raised.


.. _subgroups_rsmtool:

subgroups *(Optional)*
""""""""""""""""""""""
A list of column names indicating grouping variables used for generating analyses specific to each of those defined subgroups. For example, ``["prompt, gender, native_language, test_country"]``. These subgroup columns need to be present in both training *and* evaluation data. If subgroups are specified, ``rsmtool`` will generate:

    - description of the data by each subgroup;
    - boxplots showing the feature distribution for each subgroup on the training set; and
    - tables and barplots showing system-human agreement for each subgroup on the evaluation set.

.. _general_sections_rsmtool:

general_sections *(Optional)*
"""""""""""""""""""""""""""""
RSMTool provides pre-defined sections for ``rsmtool`` (listed below) and, by default, all of them are included in the report. However, you can choose a subset of these pre-defined sections by specifying a list as the value for this field.

    - ``data_description``: Shows the total number of responses in training and evaluation set, along with any responses have been excluded due to non-numeric features/scores or :ref:`flag columns <flag_column_rsmtool>`.

    - ``data_description_by_group``: Shows the total number of responses in training and evaluation set for each of the :ref:`subgroups <subgroups_rsmtool>` specified in the configuration file. This section only covers the responses used to train/evaluate the model.

    - ``feature_descriptives``: Shows the descriptive statistics for all raw  feature values included in the model:

        - a table showing mean, standard deviation, min, max, correlation with human score etc.;
        - a table with percentiles and outliers; and
        - a barplot showing the number of truncated outliers for each feature.

    - ``features_by_group``: Shows boxplots with distributions of raw feature values by each of the :ref:`subgroups <subgroups_rsmtool>` specified in the configuration file.

    - ``preprocessed_features``: Shows analyses of preprocessed features:

        - histograms showing the distributions of preprocessed features values;
        - the correlation matrix between all features and the human score;
        - a barplot showing marginal and partial correlations between all features and the human score, and, optionally, response length if :ref:`length_column <length_column_rsmtool>` is specified in the config file.

    - ``dff_by_group``: Differential feature functioning by group. The plots in this section show average feature values for each of the :ref:`subgroups <subgroups_rsmtool>` conditioned on human score. 

     - ``consistency``: Shows metrics for human-human agreement, the difference ('degradation') between the human-human and human-system agreement, and the disattenuated human-machine correlations. This notebook is only generated if the config file specifies :ref:`second_human_score_column <second_human_score_column_rsmtool>`

    - ``model``: Shows the parameters of the learned regression model. For linear models, it also includes the standardized and relative coefficients as well as model diagnostic plots.

    - ``evaluation``: Shows the standard set of evaluations recommended for scoring models on the evaluation data:

       - a table showing system-human association metrics;
       - the confusion matrix; and
       - a barplot showing the distributions for both human and machine scores.

    - ``evaluation_by_group``: Shows barplots with the main evaluation metrics by each of the subgroups specified in the configuration file.

    - ``true_score_evaluation``: evaluation of system scores against the true scores estimated according to test theory. The notebook shows:
    
        - variance of human scores for single and double-scored responses;
        - variance of system scores and proportional reduction in mean squared error (PRMSE) when predicting true score with system score.

    - ``pca``: Shows the results of principal components analysis on the processed feature values:

        - the principal components themselves;
        - the variances; and
        - a Scree plot.

    - ``intermediate_file_paths``: Shows links to all of the intermediate files that were generated while running the experiment.

    - ``sysinfo``: Shows all Python packages along with versions installed in the current environment while generating the report.

.. _custom_sections_rsmtool:

custom_sections *(Optional)*
""""""""""""""""""""""""""""
A list of custom, user-defined sections to be included into the final report. These are IPython notebooks (``.ipynb`` files) created by the user.  The list must contains paths to the notebook files, either absolute or relative to the configuration file. All custom notebooks have access to some :ref:`pre-defined variables <custom_notebooks>`.

.. _special_sections_rsmtool:

special_sections *(Optional)*
"""""""""""""""""""""""""""""
A list specifying special ETS-only sections to be included into the final report. These sections are available *only* to ETS employees via the `rsmextra` package.

section_order *(Optional)*
""""""""""""""""""""""""""
A list containing the order in which the sections in the report should be generated. Any specified order must explicitly list:

    1. Either *all* pre-defined sections if a value for the :ref:`general_sections <general_sections_rsmtool>` field is not specified OR the sections specified using :ref:`general_sections <general_sections_rsmtool>`, and

    2. *All* custom section names specified using :ref:`custom_ sections <custom_sections_rsmtool>`, i.e., file prefixes only, without the path and without the `.ipynb` extension, and

    3. *All* special sections specified using :ref:`special_sections <special_sections_rsmtool>`.


.. _use_thumbnails_rsmtool:

use_thumbnails *(Optional)*
"""""""""""""""""""""""""""""""""""
If set to ``true``, the images in the HTML will be set to clickable thumbnails rather than full-sized images. Upon clicking the thumbnail, the full-sized images will be displayed in a separate tab in the browser. If set to ``false``, full-sized images will be displayed as usual. Defaults to ``false``.


candidate_column *(Optional)*
"""""""""""""""""""""""""""""
The name for an optional column in the training and test data containing unique candidate IDs. Candidate IDs are different from response IDs since the same candidate (test-taker) might have responded to multiple questions.

min_items_per_candidate *(Optional)*
""""""""""""""""""""""""""""""""""""
An integer value for the minimum number of responses expected from each candidate. If any candidates have fewer responses than the specified value, all responses from those candidates will be excluded from further analysis. Defaults to ``None``.

