.. _intermediate_files_rsmeval:

Intermediate files
""""""""""""""""""
Although the primary output of ``rsmeval`` is an HTML report, we also want the user to be able to conduct additional analyses outside of ``rsmeval``. To this end, all of the tables produced in the experiment report are saved as files in the format as specified by ``file_format`` parameter in the ``output`` directory. The following sections describe all of the intermediate files that are produced.

.. note::

    The names of all files begin with the ``experiment_id`` provided by the user in the experiment configuration file. In addition, the names for certain columns are set to default values in these files irrespective of what they were named in the original data files. This is because RSMEval standardizes these column names internally for convenience. These values are:

    - ``spkitemid`` for the column containing response IDs.
    - ``sc1`` for the column containing the human scores used as observed scores
    - ``sc2`` for the column containing the second human scores, if this column was specified in the configuration file.
    - ``candidate`` for the column containing candidate IDs, if this column was specified in the configuration file.

Predictions
~~~~~~~~~~~
filename: ``pred_processed``

This file contains the post-processed predicted scores: the predictions from the model are truncated, rounded, and re-scaled (if requested).

Flagged responses
~~~~~~~~~~~~~~~~~
filename: ``test_responses_with_excluded_flags``

This file contains all of the rows in input predictions file that were filtered out based on conditions specified in :ref:`flag_column <flag_column_eval>`.

.. note::

    If the predictions file contained columns with internal names such as ``sc1`` that were not actually used by ``rsmeval``, they will still be included in these files but their names will be changed to ``##name##`` (e.g. ``##sc1##``).

Excluded responses
~~~~~~~~~~~~~~~~~~
filename: ``test_excluded_responses``

This file contains all of the rows in the predictions file that were filtered out because of non-numeric or zero scores.

Response metadata
~~~~~~~~~~~~~~~~~
filename:  ``test_metadata``

This file contains the metadata columns (``id_column``,  ``subgroups`` if provided) for all rows in the predictions file that used in the evaluation.

Unused columns
~~~~~~~~~~~~~~
filename: ``test_other_columns``

This file contains all of the the columns from the input predictions file that are not present in the ``*_pred_processed`` and ``*_metadata`` files. They only include the rows that were not filtered out.

.. note::

    If the predictions file contained columns with internal names such as ``sc1`` but these columns were not actually used by ``rsmeval``, these columns will also be included into these files but their names will be changed to ``##name##`` (e.g. ``##sc1##``).

Human scores
~~~~~~~~~~~~
filename: ``test_human_scores``

This file contains the human scores, if available in the input predictions file, under a column called ``sc1`` with the response IDs under the ``spkitemid`` column.

If ``second_human_score_column`` was specfied, then it also contains the values in the predictions file from that column under a column called ``sc2``. Only the rows that were not filtered out are included.

.. note::

    If ``exclude_zero_scores``  was set to ``true`` (the default value), all zero scores in the ``second_human_score_column`` will be replaced by ``nan``.

Data composition
~~~~~~~~~~~~~~~~
filename: ``data_composition``

This file contains the total number of responses in the input predictions file. If applicable, the table will also include the number of different subgroups.

Excluded data composition
~~~~~~~~~~~~~~~~~~~~~~~~~
filenames: ``test_excluded_composition``

This file contains the composition of the set of excluded responses, e.g., why were they excluded and how many for each such exclusion.

Subgroup composition
~~~~~~~~~~~~~~~~~~~~
filename: ``data_composition_by_<SUBGROUP>``

There will be one such file for each of the specified subgroups and it contains the total number of responses in that subgroup.

Evaluation metrics
~~~~~~~~~~~~~~~~~~
- ``eval``:  This file contains the descriptives for predicted and human scores (mean, std.dev etc.) as well as the association metrics (correlation, quadartic weighted kappa, SMD etc.) for the raw as well as the post-processed scores.

- ``eval_by_<SUBGROUP>``: the same information as in `*_eval.csv` computed separately for each subgroup. However, rather than SMD, a difference of standardized means (DSM) will be calculated using z-scores.

- ``eval_short`` -  a shortened version of ``eval`` that contains specific descriptives for predicted and human scores (mean, std.dev etc.) and association metrics (correlation, quadartic weighted kappa, SMD etc.) for specific score types chosen based on recommendations by Williamson (2012). Specifically, the following columns are included (the ``raw`` or ``scale`` version is chosen depending on the value of the ``use_scaled_predictions`` in the configuration file).

    - h_mean
    - h_sd
    - corr
    - sys_mean [raw/scale trim]
    - sys_sd [raw/scale trim]
    - SMD [raw/scale trim]
    - adj_agr [raw/scale trim_round]
    - exact_agr [raw/scale trim_round]
    - kappa [raw/scale trim_round]
    - wtkappa [raw/scale trim]
    - sys_mean [raw/scale trim_round]
    - sys_sd [raw/scale trim_round]
    - SMD [raw/scale trim_round]
    - R2 [raw/scale trim]
    - RMSE [raw/scale trim]

- ``score_dist``: the distributions of the human scores and the rounded raw/scaled predicted scores, depending on the value of ``use_scaled_predictions``.

- ``confMatrix``: the confusion matrix between the the human scores and the rounded raw/scaled predicted scores, depending on the value of ``use_scaled_predictions``.

.. note::

    Please note that for raw scores, SMD values are likely to be affected by possible differences in scale.


Human-human Consistency
~~~~~~~~~~~~~~~~~~~~~~~
These files are created only if a second human score has been made available via the ``second_human_score_column`` option in the configuration file.

- ``consistency``: contains descriptives for both human raters as well as the agreement metrics between their ratings.

- ``consistency_by_<SUBGROUP>``: contains the same metrics as in ``consistency`` file computed separately for each group. However, rather than SMD, a difference of standardized means (DSM) will be calculated using z-scores.

- ``degradation``:  shows the differences between human-human agreement and machine-human agreement for all association metrics and all forms of predicted scores.

Evaluations based on test theory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- ``disattenuated_correlations``: shows the correlation between human-machine scores, human-human scores, and the disattenuated human-machine correlation computed as human-machine correlation divided by the square root of human-human correlation.

- ``disattenuated_correlations_by_<SUBGROUP>``: contains the same metrics as in ``disattenuated_correlations`` file computed separately for each group. 

- ``true_score_eval``: evaluations of system scores against estimated true score. Contains total counts of single and double-scored response, variances for human and system scores for these sets of responses, and mean squared error (MSE) and proportional reduction in mean squared error (PRMSE) when predicting true score using system score. 

Additional fairness analyses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These files contain the results of additional fairness analyses suggested in suggested in `Loukina, Madnani, & Zechner, 2019 <https://www.aclweb.org/anthology/W19-4401/>`_. 

- ``<METRICS>_by_<SUBGROUP>.ols``: a serialized object of type ``pandas.stats.ols.OLS`` containing the fitted model for estimating the variance attributed to a given subgroup membership for a given metric. The subgroups are defined by the :ref:`configuration file<subgroups_eval>`. The metrics are ``osa`` (overall score accuracy), ``osd`` (overall score difference), and ``csd`` (conditional score difference). 

- ``<METRICS>_by_<SUBGROUP>_ols_summary.txt``: a text file containing a summary of the above model

- ``estimates_<METRICS>_by_<SUBGROUP>```: coefficients, confidence intervals and *p*-values estimated by the model for each subgroup.

- ``fairness_metrics_by_<SUBGROUP>``: the :math:`R^2` (percentage of variance) and *p*-values for all  models. 
