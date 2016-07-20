Using RSMTool
=============

For most users, the primary means of using RSMTool will be via the command-line utility ``rsmtool``. We refer to each run of ``rsmtool`` as an "experiment".

Input
-----

``rsmtool`` requires the following two inputs to run an experiment:

1. Data files for training and evaluation sets in ``.csv`` format. Each row should correspond to a single response and contain numeric feature values extracted for this response. In addition, there should be a column with a unique identifier (ID) for each response and a column with the human score for each response. The names for these two columns are defined in the configuration file.

2. A configuration file in ``.json`` format with paths to the training and evaluation files, the type of regressor to use, and other information required by ``rsmtool``. See :ref:`experiment configuration file <config_file>` for more details.

By default, ``rsmtool`` will use all of the features present in the training CSV file. If you want to use a specific set of features, you need to provide a second ``.json`` file specifying the list of features. See :ref:`feature file <feature_file>` for more details.

``rsmtool`` does the following:

preprocesses the training data (outlier truncation, normalization), learns a scoring model on the pre-processed training data, and generates a report with the description of the model and in-depth evaluation of its performance on the evaluation data.


.. _config_file:

Experiment configuration file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This file provides an overall configuration for the experiment. The file must be in ``.json`` format. Here's an example of a configuration file.

<example>

There are four required fields.

experiment_id
"""""""""""""
An identifier for the experiment that will be used to name the report and all intermediate CSV files. It can be any combination of alphanumeric values and must *not* contain spaces.

model
"""""
The machine learner you want to use to build the regressi. See [available models](available_models.md) for the list of available learners.

train_file
""""""""""
The path to the training data in ``.csv`` format. Can be absolute or relative to the location of config file.

test_file
"""""""""
The path to the evaluation data in ``.csv`` format. Can be absolute or relative to the location of config file.

description *(Optional)*
""""""""""""""""""""""""
A brief description of the experiment. This will be included in the report. The description can contain spaces and punctuation. This is blank by default.

id_column *(Optional)*
""""""""""""""""""""""
The name of the column containing the response IDs. Defaults to ``spkitemid``, i.e., if this is not specified, ``rsmtool`` will look for a column called ``spkitemid`` in the training and evaluation files.

.. _train_label_column:

train_label_column *(Optional)*
"""""""""""""""""""""""""""""""
The name for the column containing the human scores in the training data. If set to to ``fake``, fake scores will be generated using randomly sampled integers. This option may be useful if you only need descriptive statistics for the data and do not care about the other analyses. Defaults to ``sc1``.

.. _test_label_column:

test_label_column *(Optional)*
""""""""""""""""""""""""""""""
The name for the column containing the human scores in the training data. If set to to ``fake``, fake scores will be generated using randomly sampled integers. This option may be useful if you only need descriptive statistics for the data and do not care about the other analyses. Defaults to ``sc1``.

.. note::

    All responses with non-numeric values in either ``train_label_column`` or ``test_label_column`` and/or those with non-numeric values for relevant features will be automatically excluded from model training and evaluation.


features *(Optional)*
"""""""""""""""""""""
The path to the ``.json`` file containing the list of features and transformations, if any . Can be absolute or relative to the location of config file.

.. note::

    By default, ``rsmtool`` uses *all* columns in the training file as features except for the ones with the following names: ``spkitemid``, ``spkitemlab``, ``itemType``, `r1`, `r2`, `score`, `sc`, `sc1`, `adj`, and the column names specified in the config file  (e.g. `length_column`, `subgroups` as well as `train_label_column` and `test_label_column`). The final set of features will be saved in the features/ folder.


length_column *(Optional)*
""""""""""""""""""""""""""
The name for the optional column in the training and evaluation data containing response length. If specified, length is included in the inter-feature and partial correlation analyses. Note that this field *should not* be specified if you want to use the length column as an actual feature in the model. In the latter scenario, the length column will automatically be included in the analyses, like any other feature. If you specify ``length_column`` *and* include the same column name as  a feature in the :ref:`feature file <feature_file>`, ``rsmtool`` will ignore the ``length_column`` setting. In addition, if ``length_column`` has missing values or if its standard deviation is 0 (both somewhat unlikely scenarios), ``rsmtool`` will *not* include any length-based analyses in the report.

second_human_score_column *(Optional)*
""""""""""""""""""""""""""""""""""""""
The name for an optional column in the test data containing a second human score for each response. If specified, additional information about human-human agreement and degradation will be computed and included in the report. Note that this column must contain either numbers or be empty. Non-numeric values are *not* accepted. Note also that the :ref:`exclude_zero_scores` option below will apply to this column too.

candidate_column *(Optional)*
"""""""""""""""""""""""""""""
The name for an optionalc column in the training and test data containing unique candidate IDs. Note that these are currently only used for data description.

.. _exclude_zero_scores:

exclude_zero_scores *(Optional)*
""""""""""""""""""""""""""""""""
By default, responses with human scores of 0 will be excluded from both training and evaluation set. Set this field to ``false`` if you want to keep responses with scores of 0. Defaults to ``true``.

flag_column *(Optional)*
""""""""""""""""""""""""
This field makes it possible to only use responses with particular values in a given column (e.g. only responses with a value of ``0`` in a column called ``ADVISORY``). The field takes a dictionary in Python format where the keys are the names of the columns and the values are lists of values for responses that will be used to train the model. For example, a value of ``{"ADVISORY": 0}`` will mean that ``rsmtool`` will *only* use responses for which the ``ADVISORY`` column has the value 0.  If  several conditions are specified (e.g., `` {"ADVISORY": 0, "ERROR": 0}``) only those responsess which satisfy *all* the conditions will be selected for further analysis (in this example, these will be the responses where the ``ADVISORY`` column has a value of 0 *and* the ``ERROR`` column has a value of 0). Defaults to ``None``.

min_items_per_candidate *(Optional)*
""""""""""""""""""""""""""""""""""""
An integer value for the minimal number of items expected from each candidate. If any candidates have less than the specified minimal number of responses left for analysis after applying all filters, all responses from such candidates will be excluded listwise from further analysis. Defaults to ``None``.



`feature_subset_file`: a master file which lists all features that should be used for feature selection. The file should be in .csv format with features listed in a column named `Feature`. It can also optionally give the expected correlation between each feature and human score. This option is only meaningful in combination with `feature_subset` or `sign` below.
Default: None

The feature list can be further constrained by using `feature_prefix` and `feature_subset`. These fields are mutually exclusive and cannot be used in the same experiment.

`feature_subset`: The supplied feature file can specify feature subsets. These should be defined as columns in `feature_file` where the name of the column is the name of the subset and each feature is assigned 1 (included into the subset) or 0 (not included into the subset). Only one subset can be specified for each experiment.

`feature_prefix`: The feature subset can also be specified by a common prefix separated by `\t`. For example, `feature_subset: 1gram, 2gram` will create a model based only on features named 1gram\t* and 2gram\t*. Several subsets can be separated by commas.

`select_transformations`:  `true`/`false`. If this option is set to `true` the system will select the most suitable transformation based on best correlation with human score. Note that `inv` is never used for features with positive values.
Default: `false`

`sign`: the guidelines to scoring models require that all coefficients in the model are positive and all features have positive correlation with human score. It is possible to specify the expected correlation for each feature in `feature_subset_file`. In this case the features with expected negative correlation will be multiplied by -1 before adding them to the model. To use this option the `feature_subset_file` must contain a column named `Sign_X` where `X` is the value of `sign` field. This column can only takes `-` or `+`.


### Score post-processing

`trim_min`: single numeric value for the lowest possible machine score. This value will be used to compute trimmed (bound) machine scores.
Default: the lowest observed human score in the training data or 1 if there are no numeric human scores.

`trim_max`: single numeric value for the highest possible machine score. This value will be used to compute trimmed (bound) machine scores.
Default: the highest observed human score in the training data or 10 if there are no numeric human scores.

`use_scaled_predictions`: `true` if you want to use scaled machine scores for the in-depth evaluations (confusion matrices, score distribution, per-prompt evaluation). Omit this field if you want to use raw scores for these evaluations. Main evaluation metrics is always computed for both scaled and raw scores.
Default: `false`

### Subgroup analysis

`subgroups`: a list of grouping variables for generating analyses by prompt or subgroup analyses. For example, `"prompt, gender, native_language, test_country"`. These subgroup columns need to be present in both training and evaluation data. If subgroups are specified, `rsmtool` will generate: (1) description of the data by group; (2) boxplots showing feature distribution for each subgroup on the training set; and (3) tables and barplots showing system-human agreement for each subgroup on the evaluation set.
Default: no subgroups specified

### Report generation

`general_sections`: a list of general sections to be included into the final report.
See [report_sections](report_sections.md) for the list of available sections.
Default: all sections available for `rsmtool`.

`special_sections`: a list of special sections to be included into the final report. These are the sections available to all local users via `rsmextra` package. See the documentation to `rsmextra` for the list of available sections.
Default: no special sections.

`custom_sections`: a list of custom user-defined sections (`.ipynb` files) to be included into the final report. These are the notebooks created by the user. Note that the list must contains paths to the `.ipynb` files, either absolute or relative to the config file. These notebooks have access to all of the information as described in [new_notebooks](new_notebooks.md).
Default: no custom sections.

`section_order`: a list containing the order in which the sections in the report should be generated. Note that 'section_order' must list: (a) either *all* of appropriate general sections appropriate for `rsmtool`, or a subset specified using 'sections', and (b) *all* sections specified under 'special_sections', and (c) *all* 'custom_sections' names (file name only, without the path and `.ipynb` extension).



.. _feature_file:

Feature file
^^^^^^^^^^^^


Output
------


Most common use cases:

- Train and evaluate a new scoring model for new data

- Re-train an existing model on new data

- Evaluate the model performance after adding a new feature

- Generate descriptive statistics for feature distributions and correlations

`rsmtool` contains a series of in-built models and also supports all regressors implemented in [SKLL](http://skll.readthedocs.org/en/latest/run_experiment.html#learners)(see [available models](available_models.md) for the full list).
