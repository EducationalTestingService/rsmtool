.. _config_file_rsmeval:

Experiment configuration file
"""""""""""""""""""""""""""""

This is a file in ``.json`` format that provides overall configuration options for an ``rsmeval`` experiment. Here's an `example configuration file <https://github.com/EducationalTestingService/rsmtool/blob/master/examples/rsmeval/config_rsmeval.json>`_ for ``rsmeval``. 

.. note:: To make it easy to get started with  ``rsmeval``, we provide a way to **automatically generate** configurations file both interactively as well as non-interactively. Novice users will find interactive generation more helpful while more advanced users will prefer non-interactive generation. See :ref:`this page <autogen_configuration>` for more details.

Next, we describe all of the ``rsmeval`` configuration fields in detail. There are four required fields and the rest are all optional. We first describe the required fields and then the optional ones (sorted alphabetically).

experiment_id
~~~~~~~~~~~~~
An identifier for the experiment that will be used to name the report and all :ref:`intermediate files <intermediate_files_rsmeval>`. It can be any combination of alphanumeric values, must *not* contain spaces, and must *not* be any longer than 200 characters.

predictions_file
~~~~~~~~~~~~~~~~
The path to the file with predictions to evaluate. The file should be in one of the :ref:`supported formats <input_file_format>`. Each row should correspond to a single response and contain the predicted and observed scores for this response. In addition, there should be a column with a unique identifier (ID) for each response. The path can be absolute or relative to the location of the configuration file.

system_score_column
~~~~~~~~~~~~~~~~~~~
The name for the column containing the scores predicted by the system. These scores will be used for evaluation.

trim_min
~~~~~~~~
The single numeric value for the lowest possible integer score that the machine should predict. This value will be used to compute the floor value for :ref:`trimmed (bound) <score_postprocessing>` machine scores as ``trim_min`` - ``trim_tolerance``.

trim_max
~~~~~~~~
The single numeric value for the highest possible integer score that the machine should predict. This value will be used to compute the ceiling value for :ref:`trimmed (bound) <score_postprocessing>` machine scores as ``trim_max`` + ``trim_tolerance``.

.. note::

    Although the ``trim_min`` and ``trim_max`` fields are optional for ``rsmtool``, they are *required* for ``rsmeval``.

candidate_column *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The name for an optional column in prediction file containing unique candidate IDs. Candidate IDs are different from response IDs since the same candidate (test-taker) might have responded to multiple questions.

.. _custom_sections_rsmeval:

custom_sections *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A list of custom, user-defined sections to be included into the final report. These are IPython notebooks (``.ipynb`` files) created by the user.  The list must contains paths to the notebook files, either absolute or relative to the configuration file. All custom notebooks have access to some :ref:`pre-defined variables <custom_notebooks>`.

description *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~
A brief description of the experiment. This will be included in the report. The description can contain spaces and punctuation. It's blank by default.

.. _exclude_zero_scores_eval:

exclude_zero_scores *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
By default, responses with human scores of 0 will be excluded from evaluations. Set this field to ``false`` if you want to keep responses with scores of 0. Defaults to ``true``.

.. _file_format_eval:

file_format *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~
The format of the :ref:`intermediate files <intermediate_files_rsmeval>`. Options are ``csv``, ``tsv``, or ``xlsx``. Defaults to ``csv`` if this is not specified.

.. _flag_column_eval:

flag_column *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~
This field makes it possible to only use responses with particular values in a given column (e.g. only responses with a value of ``0`` in a column called ``ADVISORY``). The field takes a dictionary in Python format where the keys are the names of the columns and the values are lists of values for responses that will be evaluated. For example, a value of ``{"ADVISORY": 0}`` will mean that ``rsmeval`` will *only* use responses for which the ``ADVISORY`` column has the value 0. Defaults to ``None``.

.. note::

    If  several conditions are specified (e.g., ``{"ADVISORY": 0, "ERROR": 0}``) only those responses which satisfy *all* the conditions will be selected for further analysis (in this example, these will be the responses where the ``ADVISORY`` column has a value of 0 *and* the ``ERROR`` column has a value of 0).

.. note::

    When reading the values in the supplied dictionary, ``rsmeval`` treats numeric strings, floats and integers as the same value. Thus ``1``, ``1.0``, ``"1"`` and ``"1.0"`` are all treated as the ``1.0``.

.. _general_sections_rsmeval:

general_sections *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
RSMTool provides pre-defined sections for ``rsmeval`` (listed below) and, by default, all of them are included in the report. However, you can choose a subset of these pre-defined sections by specifying a list as the value for this field.

    - ``data_description``: Shows the total number of responses, along with any responses have been excluded due to non-numeric/zero scores or :ref:`flag columns <flag_column_eval>`.

    - ``data_description_by_group``: Shows the total number of responses for each of the :ref:`subgroups <subgroups_eval>` specified in the configuration file. This section only covers the responses used to evaluate the model.

    - ``consistency``: shows metrics for :ref:`human-human agreement <consistency_metrics>`, the difference ("degradation") between the human-human and human-system agreement, and the disattenuated human-machine correlations. This notebook is only generated if the config file specifies :ref:`second_human_score_column <second_human_score_column_eval>`.

    - ``evaluation``: Shows the :ref:`standard set of evaluations <observed_score_evaluation>` recommended for scoring models on the evaluation data:

       - a table showing human-system association metrics;
       - the confusion matrix; and
       - a barplot showing the distributions for both human and machine scores.

    - ``evaluation by group``: Shows barplots with the main evaluation metrics by each of the subgroups specified in the configuration file.

    - ``fairness_analyses``: Additional :ref:`fairness analyses <fairness_extra>` suggested in `Loukina, Madnani, & Zechner, 2019 <https://www.aclweb.org/anthology/W19-4401/>`_. The notebook shows:

        - percentage of variance in squared error explained by subgroup membership
        - percentage of variance in raw (signed) error explained by subgroup membership
        - percentage of variance in raw (signed) error explained by subgroup membership when controlling for human score
        - plots showing estimates for each subgroup for each model
    
    - ``true_score_evaluation``: evaluation of system scores against the true scores estimated according to test theory. The notebook shows:

        - variance of human scores for single and double-scored responses;
        - variance of system scores and proportional reduction in mean squared error (PRMSE) when predicting true score with system score.

    - ``intermediate_file_paths``: Shows links to all of the intermediate files that were generated while running the evaluation.

    - ``sysinfo``: Shows all Python packages along with versions installed in the current environment while generating the report.

.. _human_score_column_eval:

human_score_column *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The name for the column containing the human scores for each response. The values in this column will be used as observed scores. Defaults to ``sc1``.

.. note::

    All responses with non-numeric values or zeros in either ``human_score_column`` or ``system_score_column`` will be automatically excluded from evaluation. You can use :ref:`exclude_zero_scores_eval` to keep responses with zero scores.

id_column *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~
The name of the column containing the response IDs. Defaults to ``spkitemid``, i.e., if this is not specified, ``rsmeval`` will look for a column called ``spkitemid`` in the prediction file.

min_items_per_candidate *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
An integer value for the minimum number of responses expected from each candidate. If any candidates have fewer responses than the specified value, all responses from those candidates will be excluded from further analysis. Defaults to ``None``.

min_n_per_group *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A single numeric value or a dictionary with keys as the group names listed in the `subgroups` field and values as the thresholds for the groups. When specified, only groups with *at least* this number of instances will be displayed in the tables and plots contained **in the report**. Note that this parameter *only* affects the HTML report and the figures. For all analyses -- including the computation of the population parameters -- data from *all* groups will be used. In addition, the  :ref:`intermediate files <intermediate_files_rsmeval>` will still show the results for *all* groups. 

.. note::

    If you supply a dictionary, it *must* contain a key for *every* subgroup listed in `subgroups` field. If no threshold is to be applied for some of the groups, set the threshold value for this group to 0 in the dictionary. 

.. _rater_error_variance:

rater_error_variance *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:ref:`True score evaluations<true_score_evaluation>` require an estimate of rater error variance. While by default ``rsmeval`` will compute this from double-scored responses in the data, in some cases it may be necessary to compute rater error variance on a different sample. In this case this field can be used to set rater error variance to a precomputed amount which will be used for all computations. You can use :ref:`rsmtool.utils.variance_of_errors <ve_api>` to compute rater error variance outside the main evaluation pipeline. 

.. _scale_with_eval:

scale_with *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~
In many scoring applications, system scores are :ref:`re-scaled <score_postprocessing>` so that their mean and standard deviation match those of the human scores for the training data.

If you want ``rsmeval`` to re-scale the supplied predictions, you need to provide -- as the value for this field -- the path to a second file in one of the :ref:`supported formats <input_file_format>` containing the human scores and predictions of the same system on its training data. This file *must* have two columns: the human scores under the ``sc1`` column and the predicted score under the ``prediction``.

This field can also be set to ``"asis"`` if the scores are already scaled. In this case, no additional scaling will be performed by ``rsmeval`` but the report will refer to the scores as "scaled".

Defaults to ``"raw"`` which means that no-rescaling is performed and the report refers to the scores as "raw".

.. _second_human_score_column_eval:

second_human_score_column *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The name for an optional column in the test data containing a second human score for each response. If specified, additional information about human-human agreement and degradation will be computed and included in the report. Note that this column must contain either numbers or be empty. Non-numeric values are *not* accepted. Note also that the :ref:`exclude_zero_scores_eval` option below will apply to this column too.

.. note::

    You do not need to have second human scores for *all* responses to use this option. The human-human agreement statistics will be computed as long as there is at least one response with numeric value in this column. For responses that do not have a second human score, the value in this column should be blank.

section_order *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~
A list containing the order in which the sections in the report should be generated. Any specified order must explicitly list:

    1. Either *all* pre-defined sections if a value for the :ref:`general_sections <general_sections_rsmeval>` field is not specified OR the sections specified using :ref:`general_sections <general_sections_rsmeval>`, and

    2. *All* custom section names specified using :ref:`custom_ sections <custom_sections_rsmeval>`, i.e., file prefixes only, without the path and without the `.ipynb` extension, and

    3. *All* special sections specified using :ref:`special_sections <special_sections_rsmeval>`.

.. _special_sections_rsmeval:

special_sections *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A list specifying special ETS-only sections to be included into the final report. These sections are available *only* to ETS employees via the ``rsmextra`` package.

.. _subgroups_eval:

subgroups *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~
A list of column names indicating grouping variables used for generating analyses specific to each of those defined subgroups. For example, ``["prompt, gender, native_language, test_country"]``. These subgroup columns need to be present in the input predictions file. If subgroups are specified, ``rsmeval`` will generate:

    - tables and barplots showing human-system agreement for each subgroup on the evaluation set.

trim_tolerance *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The single numeric value that will be used to pad the trimming range specified in ``trim_min`` and ``trim_max``. This value will be used to compute the ceiling and floor values for :ref:`trimmed (bound) <score_postprocessing>` machine scores as ``trim_max`` + ``trim_tolerance`` for ceiling value and ``trim_min``-``trim_tolerance`` for floor value.
Defaults to 0.4998.

.. note::
    
    For more fine-grained control over the trimming range, you can set ``trim_tolerance`` to `0` and use ``trim_min`` and ``trim_max`` to specify the exact floor and ceiling values.  

.. _use_thumbnails_rsmeval:

use_thumbnails *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~
If set to ``true``, the images in the HTML will be set to clickable thumbnails rather than full-sized images. Upon clicking the thumbnail, the full-sized images will be displayed in a separate tab in the browser. If set to ``false``, full-sized images will be displayed as usual. Defaults to ``false``.
