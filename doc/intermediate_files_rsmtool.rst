.. _intermediate_files_rsmtool:

Intermediate files
------------------

Although the primary output of RSMTool is an HTML report, we also want the user to be able to conduct additional analyses outside of RSMTool.To this end, all of the tables produced in the experiment report are saved as files in the format as specified by ``file_format`` parameter in the ``output`` directory. The following sections describe all of the intermediate files that are produced.

.. note::

    The names of all files begin with the ``experiment_id`` provided by the user in the experiment configuration file. In addition, the names for certain columns are set to default values in these files irrespective of what they were named in the original data files. This is because RSMTool standardizes these column names internally for convenience. These values are:

    - ``spkitemid`` for the column containing response IDs.
    - ``sc1`` for the column containing the human scores used as training labels.
    - ``sc2`` for the column containing the second human scores, if this column was specified in the configuration file.
    - ``length`` for the column containing response length, if this column was specified in the configuration file.
    - ``candidate`` for the column containing candidate IDs, if this column was specified in the configuration file.


.. _rsmtool_feature_values:

Feature values
^^^^^^^^^^^^^^
filenames: ``train_features``, ``test_features``, ``train_preprocessed_features``, ``test_preprocessed_features``

These files contain the raw and pre-processed feature values for the training and evaluation sets. They include *only* includes the rows that were used for training/evaluating the models after filtering. For models with feature selection, these files *only* include the features that ended up being included in the model.

.. note::

    By default RSMTool filters out non-numeric feature values and non-numeric/zero human scores from both the training and evaluation sets. Zero scores can be kept by setting the `exclude_zero_scores` to `false`.

.. _rsmtool_flagged_responses:

Flagged responses
^^^^^^^^^^^^^^^^^
filenames: ``train_responses_with_excluded_flags``, ``test_responses_with_excluded_flags``

These files contain all of the rows in the training and evaluation sets that were filtered out based on conditions specified in :ref:`flag_column <flag_column_rsmtool>`.

.. note::

    If the training/evaluation files contained columns with internal names such as ``sc1`` or ``length`` but these columns were not actually used by ``rsmtool``, these columns will also be included into these files but their names will be changed to ``##name##`` (e.g. ``##sc1##``).

Excluded responses
^^^^^^^^^^^^^^^^^^
filenames: ``train_excluded_responses``, ``test_excluded_responses``

These files contain all of the rows in the training and evaluation sets that were filtered out because of feature values or scores. For models with feature selection, these files *only* include the features that ended up being included in the model.

Response metadata
^^^^^^^^^^^^^^^^^
filenames: ``train_metadata``, ``test_metadata``

These files contain the metadata columns (``id_column``,  ``subgroups`` if provided) for the rows in the training and evaluation sets that were *not* excluded for some reason.

.. _rsmtool_unused_columns:

Unused columns
^^^^^^^^^^^^^^
filenames: ``train_other_columns``, ``test_other_columns``

These files contain all of the the columns from the original features files that are not present in the ``*_feature`` and ``*_metadata`` files. They only include the rows from the training and evaluation sets that were not filtered out.

.. note::

    If the training/evaluation files contained columns with internal names such as ``sc1`` or ``length`` but these columns were not actually used by ``rsmtool``, these columns will also be included into these files but their names will be changed to ``##name##`` (e.g. ``##sc1##``).

Response length
^^^^^^^^^^^^^^^
filename: ``train_response_lengths``

If `length_column` is specified, then this file contains the values from that column for the training data under a column called ``length`` with the response IDs under the ``spkitemid`` column.

Human scores
^^^^^^^^^^^^
filename: ``test_human_scores``

This file contains the human scores for the evaluation data under a column called ``sc1`` with the response IDs under the ``spkitemid`` column. If ``second_human_score_column`` was specfied, then it also contains the values from that column under a column called ``sc2``. Only the rows that were not filtered out are included.

.. note::

    If ``exclude_zero_scores``  was set to ``true`` (the default value), all zero scores in the ``second_human_score_column`` will be replaced by ``nan``.

Data composition
^^^^^^^^^^^^^^^^
filename: ``data_composition``

This file contains the total number of responses in training and evaluation set and the number of overlapping responses. If applicable, the table will also include the number of different subgroups for each set.

Excluded data composition
^^^^^^^^^^^^^^^^^^^^^^^^^
filenames: ``train_excluded_composition``, ``test_excluded_composition``

These files contain the composition of the set of excluded responses for the training and evaluation sets, e.g., why were they excluded and how many for each such exclusion.

Missing features
^^^^^^^^^^^^^^^^
filename: ``train_missing_feature_values``

This file contains the total number of non-numeric values for each feature. The counts in this table are based only on those responses that have a numeric human score in the training data.

Subgroup composition
^^^^^^^^^^^^^^^^^^^^
filename: ``data_composition_by_<SUBGROUP>``

There will be one such file for each of the specified subgroups and it contains the total number of responses in  that subgroup in both the training and evaluation sets.

Feature descriptives
^^^^^^^^^^^^^^^^^^^^^
filenames: ``feature_descriptives``, ``feature_descriptivesExtra``

The first file contains the main descriptive statistics (mean,std. dev., correlation with human score etc.) for all features included in the final model. The second file contains percentiles, mild, and extreme outliers for the same set of features. The values in both files are computed on raw feature values before pre-processing.

Feature outliers
^^^^^^^^^^^^^^^^
filename: ``feature_outliers``

This file contains the number and percentage of outlier values truncated to [MEAN-4\*SD, MEAN+4\*SD] during feature pre-processing for each feature included in the final model.

Inter-feature and score correlations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
filenames: ``cors_orig``, ``cors_processed``

The first file contains the pearson correlations between each pair of (raw) features and between each (raw) feature and the human score. The second file is the same but with the pre-processed feature values instead of the raw values.

Marginal and partial correlations with score
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
filenames: ``margcor_score_all_data``, ``pcor_score_all_data``, ```pcor_score_no_length_all_data``

The first file contains the marginal correlations between each pre-processed feature and human score. The second file contains the partial correlation between each pre-processed feature and human score after controlling for all other features. The third file contains the partial correlations between each pre-processed feature and human score after controlling for response length, if ``length_column`` was specified in the configuration file.

Marginal and partial correlations with length
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
filenames: ``margcor_length_all_data``, ``pcor_length_all_data``

The first file contains the marginal correlations between each pre-processed feature and response length, if ``length_column`` was specified. The second file contains the partial correlations between each pre-processed feature and response length after controlling for all other features, if ``length_column`` was specified in the configuration file.

Principal components analyses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
filenames: ``pca``, ``pcavar``

The first file contains the the results of a Principal Components Analysis (PCA) using pre-processed feature values from the training set and its singular value decomposition. The second file contains the eigenvalues and variance explained by each component.

Various correlations by subgroups
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Each of following files may be produced for every subgroup, assuming all other information was also available.

- ``margcor_score_by_<SUBGROUP>``: the marginal correlations between each pre-processed feature and human score, computed separately for the subgroup.

- ``pcor_score_by_<SUBGROUP>``: the partial correlations between pre-processed features and human score after controlling for all other features, computed separately for the subgroup.

- ``pcor_score_no_length_by_<SUBGROUP>``: the partial correlations between each pre-processed feature and human score after controlling for response length (if available), computed separately for the subgroup.

- ``margcor_length_by_<SUBGROUP>``: the marginal correlations between each feature and response length (if available), computed separately for each subgroup.

- ``pcor_length_by_<SUBGROUP>``: partial correlations between each feature and response length (if available) after controlling for all other features, computed separately for each subgroup.

.. note::

    All of the feature descriptive statistics, correlations (including those for subgroups), and PCA are computed *only* on the training set.

Model information
^^^^^^^^^^^^^^^^^

.. _rsmtool_feature_csv:

- ``feature``: :ref:`pre-processing parameters <preprocessing_parameters>` for all features used in the model.

- ``coefficients``: model coefficients and intercept (for :ref:`built-in models <builtin_models>` only).

- ``coefficients_scaled``: scaled model coefficients and intercept (linear models only). Although RSMTool generates scaled scores by scaling the predictions of the model, it is also possible to achieve the same result by scaling the coefficients instead. This file shows those scaled coefficients.

.. _rsmtool_betas_csv:

- ``betas``: standardized and relative coefficients (for built-in models only).

- ``model_fit``: R squared and adjusted R squared computed on the training set. Note that these values are always computed on raw predictions without any trimming or rounding.

- ``.model``: the serialized SKLL ``Learner`` object containing the fitted model (before scaling the coeffcients).

- ``.ols``: a serialized object of type ``pandas.stats.ols.OLS`` containing the fitted model (for built-in models excluding ``LassoFixedLambda`` and ``PositiveLassoCV``).

- ``ols_summary.txt``: a text file containing a summary of the above model (for built-in models excluding ``LassoFixedLabmda`` and ``PositiveLassoCV``)

.. _rsmtool_postprocessing_params_csv:

- ``postprocessing_params``: the parameters for trimming and scaling predicted scores. Useful for generating predictions on new data.

.. _rsmtool_predictions:

Predictions
^^^^^^^^^^^
filenames: ``pred_processed``, ``pred_train``

The first file contains the predicted scores for the evaluation set and the second file contains the predicted scores for the responses in the training set. Both of them contain the raw scores as well as different types of post-processed scores.


.. _rsmtool_eval_files:

Evaluation metrics
^^^^^^^^^^^^^^^^^^
- ``eval``:  This file contains the descriptives for predicted and human scores (mean, std.dev etc.) as well as the association metrics (correlation, quadartic weighted kappa, SMD etc.) for the raw as well as the post-processed scores.

- ``eval_by_<SUBGROUP>``: the same information as in `*_eval.csv` computed separately for each subgroup. However, rather than SMD, a difference of standardized means (DSM) will be calculated using z-scores.

- ``eval_short`` -  a shortened version of ``eval`` that contains specific descriptives for predicted and human scores (mean, std.dev etc.) and association metrics (correlation, quadartic weighted kappa, SMD etc.) for specific score types chosen based on recommendations by Williamson (2012). Specifically, the following columns are included (the ``raw`` or ``scale`` version is chosen depending on the value of the ``use_scaled_predictions`` in the configuration file).

    - h_mean
    - h_sd
    - corr
    - sys_mean [raw/scale_trim]
    - sys_sd [raw/scale_trim]
    - SMD [raw/scale_trim]
    - adj_agr [raw/scale_trim_round]
    - exact_agr [raw/scale_trim_round]
    - kappa [raw/scale_trim_round]
    - wtkappa [raw/scale_trim]
    - sys_mean [raw/scale_trim_round]
    - sys_sd [raw/scale_trim_round]
    - SMD [raw/scale_trim_round]
    - R2 [raw/scale_trim]
    - RMSE [raw/scale_trim]

- ``score_dist``: the distributions of the human scores and the rounded raw/scaled predicted scores, depending on the value of ``use_scaled_predictions``.

- ``confMatrix``: the confusion matrix between the the human scores and the rounded raw/scaled predicted scores, depending on the value of ``use_scaled_predictions``.

.. note::

    Please note that for raw scores, SMD values are likely to be affected by possible differences in scale.

- ``true_score_eval`` - evaluation of how well system scores can predict true scores.

.. _rsmtool_consistency_files:

Human-human Consistency
^^^^^^^^^^^^^^^^^^^^^^^
These files are created only if a second human score has been made available via the ``second_human_score_column`` option in the configuration file.

- ``consistency``: contains descriptives for both human raters as well as the agreement metrics between their ratings.


- ``consistency_by_<SUBGROUP>``: contains the same metrics as in ``consistency`` file computed separately for each group. However, rather than SMD, a difference of standardized means (DSM) will be calculated using z-scores.

- ``degradation``:  shows the differences between human-human agreement and machine-human agreement for all association metrics and all forms of predicted scores.


.. _rsmtool_true_score_eval:

Evaluations based on test theory
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``disattenuated_correlations``: shows the correlation between human-machine scores, human-human scores, and the disattenuated human-machine correlation computed as human-machine correlation divided by the square root of human-human correlation.

- ``disattenuated_correlations_by_<SUBGROUP>``: contains the same metrics as in ``disattenuated_correlations`` file computed separately for each group.

- ``true_score_eval``: evaluations of system scores against estimated true score. Contains total counts of single and double-scored response, variance of human rater error, estimated true score variance, and mean squared error (MSE) and proportional reduction in mean squared error (PRMSE) when predicting true score using system score.

.. _rsmtool_fairness_eval:

Additional fairness analyses
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These files contain the results of additional fairness analyses suggested in suggested in `Loukina, Madnani, & Zechner, 2019 <https://www.aclweb.org/anthology/W19-4401/>`_.

- ``<METRICS>_by_<SUBGROUP>.ols``: a serialized object of type ``pandas.stats.ols.OLS`` containing the fitted model for estimating the variance attributed to a given subgroup membership for a given metric. The subgroups are defined by the :ref:`configuration file<subgroups_rsmtool>`. The metrics are ``osa`` (overall score accuracy), ``osd`` (overall score difference), and ``csd`` (conditional score difference).

- ``<METRICS>_by_<SUBGROUP>_ols_summary.txt``: a text file containing a summary of the above model

- ``estimates_<METRICS>_by_<SUBGROUP>```: coefficients, confidence intervals and *p*-values estimated by the model for each subgroup.

- ``fairness_metrics_by_<SUBGROUP>``: the :math:`R^2` (percentage of variance) and *p*-values for all  models.
