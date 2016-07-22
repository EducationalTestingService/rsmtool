.. _intermediate_files_rsmeval:

Intermediate CSV files
""""""""""""""""""""""
Although the primary output of ``rsmeval`` is an HTML report, we also want the user to be able to conduct additional analyses outside of ``rsmeval``. To this end, all of the tables produced in an the experiment report are saved as ``.csv`` files in the ``output`` directory. The following sections describe all of the intermediate files that are produced.

.. note::

    The names of all files begin with the ``experiment_id`` provided by the user in the experiment configuration file. In addition, the names for certain columns are set to default values in these files irrespective of what they were named in the original ``.csv`` files. This is because RSMEval standardizes these column names internally for convenience. These values are:

    - ``spkitemid`` for the column containing response IDs.
    - ``sc1`` for the column containing the human scores used as observed scores
    - ``sc2`` for the column containing the second human scores, if this column was specified in the configuration file.
    - ``candidate`` for the column containing candidate IDs, if this column was specified in the configuration file.

Predictions
~~~~~~~~~~~
filename: ``pred_processed.csv``

This file contains the post-processed predicted scores: the predictions from the model are truncated, rounded, and re-scaled (if requested).

Flagged responses
~~~~~~~~~~~~~~~~~
filename: ``test_responses_with_excluded_flags.csv``

This file contains all of the rows in input predictions file that were filtered out based on conditions specified in :ref:`flag_column <flag_column_eval>`.

.. note::

    If the predictions file contained columns with internal names such as ``sc1`` that were not actually used by ``rsmeval``, they will still be included in these files but their names will be changed to ``##name##`` (e.g. ``##sc1##``).

Excluded responses
~~~~~~~~~~~~~~~~~~
filename: ``test_excluded_responses.csv``

This file contains all of the rows in the predictions file that were filtered out because of non-numeric or zero scores.

Response metadata
~~~~~~~~~~~~~~~~~
filename:  ``test_metadata.csv``

This file contains the metadata columns (``id_column``,  ``subgroups`` if provided) for all rows in the predictions file that used in the evaluation.

Unused columns
~~~~~~~~~~~~~~
filename: ``test_other_columns.csv``

This file contains all of the the columns from the input predictions file that are not present in the ``*_pred_processed.csv`` and ``*_metadata.csv`` files. They only include the rows that were not filtered out.

.. note::

    If the predictions file contained columns with internal names such as ``sc1`` but these columns were not actually used by ``rsmeval``, these columns will also be included into these files but their names will be changed to ``##name##`` (e.g. ``##sc1##``).

Human scores
~~~~~~~~~~~~
filename: ``test_human_scores.csv``

This file contains the human scores, if available in the input predictions file, under a column called ``sc1`` with the response IDs under the ``spkitemid`` column.

If ``second_human_score_column`` was specfied, then it also contains the values in the predictions file from that column under a column called ``sc2``. Only the rows that were not filtered out are included.

.. note::

    If ``exclude_zero_scores``  was set to ``true`` (the default value), all zero scores in the ``second_human_score_column`` will be replaced by ``nan``.

Data composition
~~~~~~~~~~~~~~~~
filename: ``data_composition.csv``

This file contains the total number of responses in the input predictions file. If applicable, the table will also include the number of different subgroups.

Excluded data composition
~~~~~~~~~~~~~~~~~~~~~~~~~
filenames: ``test_excluded_composition.csv``

This file contains the composition of the set of excluded responses, e.g., why were they excluded and how many for each such exclusion.

Subgroup composition
~~~~~~~~~~~~~~~~~~~~
filename: ``data_composition_by_<SUBGROUP>.csv``

There will be one such file for each of the specified subgroups and it contains the total number of responses in that subgroup.

Evaluation metrics
~~~~~~~~~~~~~~~~~~
- ``eval.csv``:  This file contains the descriptives for predicted and human scores (mean, std.dev etc.) as well as the association metrics (correlation, quadartic weighted kappa, SMD etc.) for the raw as well as the post-processed scores.

- ``eval_by_<SUBGROUP>.csv``: the same information as in `*_eval.csv` computed separately for each subgroup.

- ``eval_short.csv`` -  a shortened version of ``eval.csv`` that contains specific descriptives for predicted and human scores (mean, std.dev etc.) and association metrics (correlation, quadartic weighted kappa, SMD etc.) for specific score types chosen based on recommendations by Williamson (2012). Specifically, the following columns are included (the ``raw`` or ``scale`` version is chosen depending on the value of the ``use_scaled_predictions`` in the configuration file).

    - h_mean
    - h_sd
    - corr
    - sys_mean [raw/scale trim]
    - sys_sd [raw/scale trim]
    - SMD [raw/scale trim]
    - adj_agr [raw/scale trim_round]
    - exact_agr [raw/scale trim_round]
    - kappa [raw/scale trim_round]
    - wtkappa [raw/scale trim_round]
    - sys_mean [raw/scale trim_round]
    - sys_sd [raw/scale trim_round]
    - SMD [raw/scale trim_round]
    - R2 [raw/scale trim]
    - RMSE [raw/scale trim]

- ``score_dist.csv``: the distributions of the human scores and the rounded raw/scaled predicted scores, depending on the value of ``use_scaled_predictions``.

- ``confMatrix.csv``: the confusion matrix between the the human scores and the rounded raw/scaled predicted scores, depending on the value of ``use_scaled_predictions``.

Human-human Consistency
~~~~~~~~~~~~~~~~~~~~~~~
These files are created only if a second human score has been made available via the ``second_human_score_column`` option in the configuration file.

- ``consistency.csv``: contains descriptives for both human raters as well as the agreement metrics between their ratings.

- ``degradation.csv``:  shows the differences between human-human agreement and machine-human agreement for all association metrics and all forms of predicted scores.
