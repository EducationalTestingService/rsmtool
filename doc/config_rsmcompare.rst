.. _config_file_rsmcompare:

Experiment configuration file
"""""""""""""""""""""""""""""

This is a file in ``.json`` format that provides overall configuration options for an ``rsmcompare`` experiment. An example configuration file can be found here.

There are six required fields and the rest are all optional.

experiment_id_old
~~~~~~~~~~~~~~~~~
An identifier for the "baseline" experiment. This ID should be identical to the ``experiment_id`` used when the baseline experiment was run, whether ``rsmtool`` or ``rsmeval``. The results for this experiment will be listed first in the comparison report.

experiment_id_new
~~~~~~~~~~~~~~~~~
An identifier for the experiment with the "new" model (e.g., the model with new feature(s)). This ID should be identical to the ``experiment_id`` used when the experiment was run, whether ``rsmtool`` or ``rsmeval``. The results for this experiment will be listed first in the comparison report.

experiment_dir_old
~~~~~~~~~~~~~~~~~~
The directory with the results for the "baseline" experiment. This directory is the output directory that was used for the experiment and should contain subdirectories ``output`` and ``figure``.

experiment_dir_new
~~~~~~~~~~~~~~~~~~
The directory with the results for the experiment with the new model. This directory is the output directory that was used for the experiment and should contain subdirectories ``output`` and ``figure``.

description_old
~~~~~~~~~~~~~~~
A brief description of the "baseline" experiment. The description can contain spaces and punctuation.

description_new
~~~~~~~~~~~~~~~
A brief description of the experiment with the new model. The description can contain spaces and punctuation.

use_scaled_predictions_old *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Set to ``true`` if the "baseline" experiment used scaled machine scores for confusion matrices, score distributions, subgroup analyses, etc. Defaults to ``false``.

use_scaled_predictions_new *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Set to ``true`` if the experiment with the new model used scaled machine scores for confusion matrices, score distributions, subgroup analyses, etc. Defaults to ``false``.

subgroups *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~
A list of column names indicating grouping variables used for generating analyses specific to each of those defined subgroups.

.. note::

    In order to include subgroups analyses in the comparison report, both experiments must have been run with the same set of subgroups.

.. _general_sections_rsmcompare:

general_sections *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
RSMTool provides pre-defined sections for ``rsmcompare`` (listed below) and, by default, all of them are included in the report. However, you can choose a subset of these pre-defined sections by specifying a list as the value for this field.

    - ``feature_descriptives``: Compares the descriptive statistics for all raw feature values included in the model:

        - a table showing mean, standard deviation, min, max, correlation with human score etc.;
        - a table with percentiles and outliers; and
        - a barplot showing he number of truncated outliers for each feature.

    - ``features_by_group``: Shows boxplots for both experiments with distributions of raw feature values by each of the :ref:`subgroups <subgroups_rsmtool>` specified in the configuration file.

    - ``preprocessed_features``: Compares analyses of preprocessed features:

        - histograms showing the distributions of preprocessed features values;
        - the correlation matrix between all features and the human score;
        - a barplot showing marginal and partial correlations between all features and the human score, and, optionally, response length if :ref:`length_column <length_column_rsmtool>` is specified in the config file.

    - ``preprocessed_features_by_group``: Compares analyses of preprocessed features by subgroups: marginal and partial correlations between each feature and human score for each subgroup.

    - ``pca``: Shows the results of principal components analysis on the processed feature values:

        - the principal components themselves;
        - the variances; and
        - a Scree plot.

    - ``model``: Compares the parameters of the two regression models. For linear models, it also includes the standardized and relative coefficients as well as model diagnostic plots.

    - ``evaluation``: Compares the standard set of evaluations recommended for scoring models on the evaluation data:

       - a table showing system-human association metrics;
       - the confusion matrix; and
       - a barplot showing the distributions for both human and machine scores.

    - ``evaluation by group``: Shows barplots for both experiments with the main evaluation metrics by each of the subgroups specified in the configuration file.

    - ``sysinfo``: Shows all Python packages along with versions installed in the current environment while generating the report.

    - ``notes``: Notes explaining the terminology used in comparison reports.

.. _custom_sections_rsmcompare:

custom_sections *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A list of custom, user-defined sections to be included into the final report. These are IPython notebooks (``.ipynb`` files) created by the user.  The list must contains paths to the notebook files, either absolute or relative to the configuration file. All custom notebooks have access to some :ref:`pre-defined variables <custom_notebooks>`.

.. _special_sections_rsmcompare:

special_sections *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A list specifying special ETS-only comparison sections to be included into the final report. These sections are available *only* to ETS employees via the `rsmextra` package.

section_order *(Optional)*
~~~~~~~~~~~~~~~~~~~~~~~~~~
A list containing the order in which the sections in the report should be generated. Any specified order must explicitly list:

    1. Either *all* pre-defined sections if a value for the :ref:`general_sections <general_sections_rsmcompare>` field is not specified OR the sections specified using :ref:`general_sections <general_sections_rsmcompare>`, and

    2. *All* custom section names specified using :ref:`custom_ sections <custom_sections_rsmcompare>`, i.e., file prefixes only, without the path and without the `.ipynb` extension, and

    3. *All* special sections specified using :ref:`special_sections <special_sections_rsmcompare>`.
