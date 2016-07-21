Experiment configuration file
"""""""""""""""""""""""""""""

This is a file in ``.json`` format that provides overall configuration options for an ``rsmeval`` experiment. An example configuration file can be found here.

There are four required fields and the rest are all optional.

experiment_id
~~~~~~~~~~~~~
An identifier for the experiment that will be used to name the report and all intermediate CSV files. It can be any combination of alphanumeric values and must *not* contain spaces.

predictions_file
~~~~~~~~~~~~~~~~
The path to the file with predictions in ``.csv`` format. Can be absolute or relative to the location of config file.  Each row should correspond to a single response and contain the predicted and observed scores for this response. In addition, there should be a column with a unique identifier (ID) for each response. The path can be absolute or relative to the location of config file.

system_score_column
~~~~~~~~~~~~~~~~~~~
The name for the column containing the system scores (predicted scores). These scores will be used for evaluation. 

trim_min
~~~~~~~~
The single numeric value for the lowest possible score that the machine should predict. This value will be used to compute trimmed (bound) machine scores.

trim_max
~~~~~~~~

The single numeric value for the highest possible score that the machine should predict. This value will be used to compute trimmed (bound) machine scores. 

.. note::

    ``trim_min`` and ``trim_max`` fields are optional fields for ``rsmtool`` but are required fields for ``rsmeval``.

description *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~

A brief description of the experiment. This will be included in the report. The description can contain spaces and punctuation. This is blank by default.



id_column *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~

The name of the column containing the response IDs. Defaults to ``spkitemid``, i.e., if this is not specified, ``rsmeval`` will look for a column called ``spkitemid`` in the prediction file.


human_score_column *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The name for the column containing the human scores for each response. The values in this column will be used as observed scores. Defaults to ``sc1``. 


.. note::

    All responses with non-numeric values or zeros in either ``human_score_column`` or ``system_score_column`` will be automatically excluded from model training and evaluation. You can use :ref:`exclude_zero_scores_eval` to keep responses with zero scores. 


second_human_score_column *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The name for an optional column in the test data containing a second human score for each response. If specified, additional information about human-human agreement and degradation will be computed and included in the report. Note that this column must contain either numbers or be empty. Non-numeric values are *not* accepted. Note also that the :ref:`exclude_zero_scores_eval` option below will apply to this column too.

.. _flag_column_eval:

flag_column *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~

This field makes it possible to only use responses with particular values in a given column (e.g. only responses with a value of ``0`` in a column called ``ADVISORY``). The field takes a dictionary in Python format where the keys are the names of the columns and the values are lists of values for responses that will be used to train the model. For example, a value of ``{"ADVISORY": 0}`` will mean that ``rsmtool`` will *only* use responses for which the ``ADVISORY`` column has the value 0. Defaults to ``None``.

.. note::

    If  several conditions are specified (e.g., ``{"ADVISORY": 0, "ERROR": 0}``) only those responses which satisfy *all* the conditions will be selected for further analysis (in this example, these will be the responses where the ``ADVISORY`` column has a value of 0 *and* the ``ERROR`` column has a value of 0).

.. _exclude_zero_scores_Eval:

exclude_zero_scores *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, responses with human scores of 0 will be excluded from the analysis. Set this field to ``false`` if you want to keep responses with scores of 0. Defaults to ``true``.


scale_with *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~"

In some applications system scores are re-scaled so that their mean and standard deviation match those of human scores. If the system scores need to be re-scaled to human range, this field should contain a path to the file which contains the human scores (named ``sc1``) and system scores (named ``prediction``) that will be used to obtain the scaling parameters. This field can also be set to ``asis`` if the scores are already scaled: in this case no additional scaling will be done but the report will refer to the scores as "scaled".
Default: no additional scaling is done; the report refers to the scores as "raw".


.. _subgroups_eval:

subgroups *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~

A list of grouping variables used to generating analyses by those defined subgroups. For example, ``["prompt, gender, native_language, test_country"]``. These subgroup columns need to be present in both training *and* evaluation data.
If subgroups are specified, ``rsmeval`` will generate:

    - tables and barplots showing system-human agreement for each subgroup on the evaluation set.

.. _general_sections_eval:

general_sections *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A list specifying which sections should be included into the final report. By default, all of the sections below are included.

    - ``data_description``: Shows the total number of responses, along with any responses have been excluded due to non-numeric/zero scores or :ref:`flag columns <flag_column_eval>`.

    - ``data_description_by_group``: Shows the total number of responses for each of the :ref:`subgroups <subgroups_eval>` specified in the configuration file. This section only covers the responses used to evaluate the model.

    - ``evaluation``: Shows the standard set of evaluations recommended for scoring models on the evaluation data:

       - a table showing system-human association metrics;
       - the confusion matrix; and
       - a barplot showing the distributions for both human and machine scores.

    - ``evaluation by group``: Shows barplots with the main evaluation metrics by each of the subgroups specified in the configuration file.

    - ``sysinfo``: Shows all Python packages along with versions installed in the current environment while generating the report.

.. _custom_sections_eval:

custom_sections *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A list of custom, user-defined sections to be included into the final report. These are IPython notebooks (``.ipynb`` files) created by the user.  The list must contains paths to the notebook files, either absolute or relative to the configuration file. All custom notebooks have access to some :doc:`pre-defined variables <new_notebooks>`.

.. _special_sections_eval:

special_sections *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A list specifying special ETS-only sections to be included into the final report. These sections are available *only* to ETS employees via the ``rsmextra`` package.


section_order *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~

A list containing the order in which the sections in the report should be generated. Possible values are:

    - either *all* of :ref:`pre-defined sections <general_sections_eval>` in a specified order; OR
    - the subset of :ref:`pre-defined sections <general_sections_eval>` AND *all* :ref:`custom sections <custom_sections_eval>` names (file prefixes only, without the path and without the `.ipynb` extension) AND *all* :ref:`special sections <special_sections_eval>`, in a specified order.


candidate_column *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The name for an optional column in prediction file containing unique candidate IDs. Candidate IDs are different from response IDs since the same candidate (test-taker) might have responded to multiple questions.

min_items_per_candidate *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An integer value for the minimum number of responses expected from each candidate. If any candidates have fewer responses than the specified value, all responses from those candidates will be excluded from further analysis. Defaults to ``None``.
