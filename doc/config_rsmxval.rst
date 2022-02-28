.. _config_file_rsmxval:

Experiment configuration file
"""""""""""""""""""""""""""""

This is a file in ``.json`` format that provides overall configuration options for an ``rsmxval`` experiment. Here's an `example configuration file <https://github.com/EducationalTestingService/rsmtool/blob/main/examples/rsmxval/config_rsmxval.json>`__ for ``rsmxval``. 

.. note:: To make it easy to get started with  ``rsmxval``, we provide a way to **automatically generate** configuration files both interactively as well as non-interactively. Novice users will find interactive generation more helpful while more advanced users will prefer non-interactive generation. See :ref:`this page <autogen_configuration>` for more details.

Configuration files for ``rsmxval`` are almost identical to ``rsmtool`` configuration files with only a few differences. Next, we describe the three required ``rsmxval`` configuration fields in detail. 

experiment_id
~~~~~~~~~~~~~
An identifier for the experiment that will be used as part of the names of the reports and intermediate files produced in each of the steps. It can be any combination of alphanumeric values, must *not* contain spaces, and must *not* be any longer than 200 characters. Suffixes are added to this experiment ID by each of the steps for the reports and files they produce, i.e., ``_fold<N>`` in the per-fold ``rsmtool`` step where ``<N>`` is a two digit number, ``_evaluation`` by the ``rsmeval`` evaluation step, ``_fold_summary`` by the ``rsmsummarize`` step, and ``_model`` by the final full-data ``rsmtool`` step.

model
~~~~~
The machine learner you want to use to build the scoring model. Possible values include :ref:`built-in linear regression models <builtin_models>` as well as all of the learners available via `SKLL <https://skll.readthedocs.io/en/latest/run_experiment.html#learners>`_. With SKLL learners, you can customize the :ref:`tuning objective <skll_objective>` and also :ref:`compute expected scores as predictions <predict_expected_scores>`.

train_file
~~~~~~~~~~
The path to the training data feature file in one of the :ref:`supported formats <input_file_format>`. Each row should correspond to a single response and contain numeric feature values extracted for this response. In addition, there should be a column with a unique identifier (ID) for each response and a column with the human score for each response. The path can be absolute or relative to config file's location.

.. important:: Unlike ``rsmtool``, ``rsmxval`` does not accept an evaluation set and will raise an error if the ``test_file`` field is specified.

Next, we will describe the two optional fields that are unique to ``rsmxval``. 

folds *(Optional)*
~~~~~~~~~~~~~~~~~~
The number of folds to use for cross-validation. This should be an integer and defaults to 5. 

folds_file *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~
The path to a file containing custom, pre-specified folds to be used for cross-validation. This should be a ``.csv`` file (no other formats are accepted) and should contain only two columns: ``id`` and ``fold``. The ``id`` column should contain the same IDs of the responses that are contained in ``train_file`` above. The ``fold`` column should contain an integer representing which fold the response with the ``id`` belongs to. IDs not specified in this file will be skipped and not included in the cross-validation at all. Just like ``train_file``, this path can be absolute or relative to the config file's location. Here's an `example of a folds file containing 2 folds <https://github.com/EducationalTestingService/rsmtool/blob/main/tests/data/files/folds.csv>`__. 

.. note:: If *both* ``folds_file`` and ``folds`` are specified, then the former will take precedence unless it contains a non-existent path.

In addition to the fields described so far, an ``rsmxval`` configuration file also accepts the following optional fields used by ``rsmtool``:

- ``candidate_column``
- ``description``
- ``exclude_zero_scores``
- ``feature_subset``
- ``feature_subset_file``
- ``features``
- ``file_format``
- ``flag_column``
- ``flag_column_test``
- ``id_column``
- ``length_column``
- ``min_items_per_candidate``
- ``min_n_per_group``
- ``predict_expected_scores``
- ``rater_error_variance``
- ``second_human_score_column``
- ``select_transformations``
- ``sign``
- ``skll_fixed_parameters``
- ``skll_objective``
- ``standardize_features``
- ``subgroups``
- ``train_label_column``
- ``trim_max``
- ``trim_min``
- ``trim_tolerance``
- ``use_scaled_predictions``
- ``use_thumbnails``
- ``use_truncation_thresholds``

Please refer to these fields' descriptions on the page describing the :ref:`rsmtool configuration file <config_file_rsmtool>`.
