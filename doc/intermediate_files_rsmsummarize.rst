.. _intermediate_files_rsmsummarize:

Intermediate CSV files
""""""""""""""""""""""

Although the primary output of RSMSummarize is an HTML report, we also want the user to be able to conduct additional analyses outside of RSMTool. To this end, all of the tables produced in an RSMSummarize experiment report are saved as ``.csv`` files in the ``output`` directory. The following sections describe all of the intermediate files that are produced.

.. note::

    The names of all files begin with the ``summary_id`` provided by the user in the experiment configuration file. 


Marginal and partial correlations with score
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

filenames: ``margcor_score_all_data.csv``, ``pcor_score_all_data.csv``, ```pcor_score_no_length_all_data.csv``

The first file contains the marginal correlations between each pre-processed feature and human score. The second file contains the partial correlation between each pre-processed feature and human score after controlling for all other features. The third file contains the partial correlations between each pre-processed feature and human score after controlling for response length, if ``length_column`` was specified in the configuration file.

Model information
^^^^^^^^^^^^^^^^^

- ``model_summary.csv``

This file contains the main information about the models included into the report including: 

    - Total number of features
    - Total number of features with non-negative coefficients
    - The learner
    - The label used to train the model

- ``betas.csv``: standardized coefficients (for built-in models only).

- ``model_fit.csv``: R squared and adjusted R squared computed on the training set. Note that these values are always computed on raw predictions without any trimming or rounding.


.. note::
    If the report includes a combination of ``rsmtool`` and ``rsmeval`` experiments, the summary tables with model information will only include ``rsmtool`` experiments since no model information is available for ``rsmeval`` experiments.


Evaluation metrics
^^^^^^^^^^^^^^^^^^

- ``eval_short.csv`` - descriptives for predicted and human scores (mean, std.dev etc.) and association metrics (correlation, quadartic weighted kappa, SMD etc.) for specific score types chosen based on recommendations by Williamson (2012). Specifically, the following columns are included (the ``raw`` or ``scale`` version is chosen depending on the value of the ``use_scaled_predictions`` in the configuration file).

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
