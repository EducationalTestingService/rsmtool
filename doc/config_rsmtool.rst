.. _config_file:

Experiment configuration file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is a file in ``.json`` format that provides overall configuration options for an ``rsmtool`` experiment. An example configuration file can be found here.

There are four required fields and rest are all optional.

experiment_id
"""""""""""""
An identifier for the experiment that will be used to name the report and all intermediate CSV files. It can be any combination of alphanumeric values and must *not* contain spaces.

model
"""""
The machine learner you want to use to build the scoring model. Possible values include :ref:`built-in linear regression models <builtin_models>` as well as all of the regressors available via `SKLL <http://skll.readthedocs.io/en/latest/run_experiment.html#learners>`_.

train_file
""""""""""
The path to the training data feature file in ``.csv`` format. Each row should correspond to a single response and contain numeric feature values extracted for this response. In addition, there should be a column with a unique identifier (ID) for each response and a column with the human score for each response. Can be absolute or relative to the location of config file.

test_file
"""""""""
The path to the evaluation data feature file in ``.csv`` format. Each row should correspond to a single response and contain numeric feature values extracted for this response. In addition, there should be a column with a unique identifier (ID) for each response and a column with the human score for each response. Can be absolute or relative to the location of config file.

description *(Optional)*
""""""""""""""""""""""""
A brief description of the experiment. This will be included in the report. The description can contain spaces and punctuation. This is blank by default.

features *(Optional)*
"""""""""""""""""""""
By default, ``rsmtool`` will use all of the features present in the training and evaluation CSV files. If you want to use a specific set of features, you need to provide a second ``.json`` file specifying the list of features. The value for this field is the path to this ``.json`` file. It can be absolute or relative to the location of config file.

.. note::

    See :doc:`selecting features <feature_selection>` for more details.

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

.. _length_column:

length_column *(Optional)*
""""""""""""""""""""""""""
The name for the optional column in the training and evaluation data containing response length. If specified, length is included in the inter-feature and partial correlation analyses. Note that this field *should not* be specified if you want to use the length column as an actual feature in the model. In the latter scenario, the length column will automatically be included in the analyses, like any other feature. If you specify ``length_column`` *and* include the same column name as  a feature in the :ref:`feature file <feature_file>`, ``rsmtool`` will ignore the ``length_column`` setting. In addition, if ``length_column`` has missing values or if its standard deviation is 0 (both somewhat unlikely scenarios), ``rsmtool`` will *not* include any length-based analyses in the report.

second_human_score_column *(Optional)*
""""""""""""""""""""""""""""""""""""""
The name for an optional column in the test data containing a second human score for each response. If specified, additional information about human-human agreement and degradation will be computed and included in the report. Note that this column must contain either numbers or be empty. Non-numeric values are *not* accepted. Note also that the :ref:`exclude_zero_scores` option below will apply to this column too.

.. _flag_column:

flag_column *(Optional)*
""""""""""""""""""""""""
This field makes it possible to only use responses with particular values in a given column (e.g. only responses with a value of ``0`` in a column called ``ADVISORY``). The field takes a dictionary in Python format where the keys are the names of the columns and the values are lists of values for responses that will be used to train the model. For example, a value of ``{"ADVISORY": 0}`` will mean that ``rsmtool`` will *only* use responses for which the ``ADVISORY`` column has the value 0. Defaults to ``None``.

.. note::

    If  several conditions are specified (e.g., ``{"ADVISORY": 0, "ERROR": 0}``) only those responsess which satisfy *all* the conditions will be selected for further analysis (in this example, these will be the responses where the ``ADVISORY`` column has a value of 0 *and* the ``ERROR`` column has a value of 0).

.. _exclude_zero_scores:

exclude_zero_scores *(Optional)*
""""""""""""""""""""""""""""""""
By default, responses with human scores of 0 will be excluded from both training and evaluation set. Set this field to ``false`` if you want to keep responses with scores of 0. Defaults to ``true``.

trim_min *(Optional)*
"""""""""""""""""""""
The single numeric value for the lowest possible score that the machine should predict. This value will be used to compute trimmed (bound) machine scores. Defaults to the lowest observed human score in the training data or 1 if there are no numeric human scores available.

trim_max *(Optional)*
"""""""""""""""""""""
The single numeric value for the highest possible score that the machine should predict. This value will be used to compute trimmed (bound) machine scores. Defaults to the highest observed human score in the training data or 10 if there are no numeric human scores available.

use_scaled_predictions *(Optional)*
"""""""""""""""""""""""""""""""""""
If set to ``true``, certain evaluations (confusion matrices, score distributions, subgroup analyses) will use the scaled machine scores. If set to ``false``, these evaluations will use the raw machine scores. Defaults to ``false``.

.. note::

    All evaluation metrics (e.g., kappa and pearson correlation) are automatically computed for *both* scaled and raw scores.


.. _subgroups:

subgroups *(Optional)*
""""""""""""""""""""""
A list of grouping variables used to generating analyses by those defined subgroups. For example, ``["prompt, gender, native_language, test_country"]``. These subgroup columns need to be present in both training *and* evaluation data.
If subgroups are specified, ``rsmtool`` will generate:

    - description of the data by each subgroup;
    - boxplots showing the feature distribution for each subgroup on the training set; and
    - tables and barplots showing system-human agreement for each subgroup on the evaluation set.

.. _general_sections:

general_sections *(Optional)*
"""""""""""""""""""""""""""""
A list specifying which sections should be included into the final report. By default, all of the sections below are included.

    - ``data_description``: Shows the total number of responses in training and evaluation set, along with any responses have been excluded due to non-numeric features/scores or :ref:`flag columns <flag_column>`.

    - ``data_description_by_group``: Shows the total number of responses in training and evaluation set for each of the :ref:`subgroups <subgroups>` specified in the configuration file. This section only covers the responses used to train/evaluate the model.

    - ``feature_descriptives``: Shows the descriptive statistics for all raw  feature values included in the model:

        - a table showing mean, standard deviation, min, max, correlation with human score etc.;
        - a table with percentiles and outliers; and
        - a barplot showing he number of truncated outliers for each feature.

    - ``features_by_group``: Shows boxplots with distributions of raw feature values by each of the :ref:`subgroups <subgroups>` specified in the configuration file.

    - ``preprocessed_features``: Shows analyses of preprocessed features:

        - histograms showing the distributions of preprocessed features values;
        - the correlation matrix between all features and the human score;
        - a barplot showing marginal and partial correlations between all features and the human score, and, optionally, response length if :ref:`length_column <length_column>` is specified in the config file.

    - ``pca``: Shows the results of principal components analysis on the processed feature values:

        - the principal components themselves;
        - the variances; and
        - a Scree plot.

    - ``model``: Shows the parameters of the learned regression model. For linear models, it also includes the standardized and relative coefficients as well as model diagnostic plots.

    - ``evaluation``: Shows the standard set of evaluations recommended for scoring models on the evaluation data:

       - a table showing system-human association metrics;
       - the confusion matrix; and
       - a barplot showing the distributions for both human and machine scores.

    - ``evaluation by group``: Shows barplots with the main evaluation metrics by each of the subgroups specified in the configuration file.

    - ``sysinfo``: Shows all Python packages along with versions installed in the current environment while generating the report.

.. _custom_sections:

custom_sections *(Optional)*
""""""""""""""""""""""""""""
A list of custom, user-defined sections to be included into the final report. These are IPython notebooks (``.ipynb`` files) created by the user.  The list must contains paths to the notebook files, either absolute or relative to the configuration file. All custom notebooks have access to some :doc:`pre-defined variables <new_notebooks>`.

.. _special_sections:

special_sections *(Optional)*
"""""""""""""""""""""""""""""
A list specifying special ETS-only sections to be included into the final report. These sections are available *only* to ETS employees via the `rsmextra` package.


section_order *(Optional)*
""""""""""""""""""""""""""
A list containing the order in which the sections in the report should be generated. Possible values are:

    - either *all* of :ref:`pre-defined sections <general_sections>` in a specified order; OR
    - the subset of :ref:`pre-defined sections <general_sections>` AND *all* :ref:`custom sections <custom_sections>` names (file prefixes only, without the path and without the `.ipynb` extension) AND *all* :ref:`special sections <special_sections>`, in a specified order.


candidate_column *(Optional)*
"""""""""""""""""""""""""""""
The name for an optional column in the training and test data containing unique candidate IDs. Candidate IDs are different from response IDs since the same candidate (test-taker) might have responded to multiple questions.

min_items_per_candidate *(Optional)*
""""""""""""""""""""""""""""""""""""
An integer value for the minimum number of responses expected from each candidate. If any candidates have fewer responses than the specified value, all responses from those candidates will be excluded from further analysis. Defaults to ``None``.
