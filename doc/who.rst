.. _who_rsmtool:

Who is RSMTool for?
===================

We expect the primary users of RSMTool to be researchers working on developing new automated scoring engines or on improving existing ones. Here's the most common scenario.

A group of researchers already *has* a set of responses such as essays or recorded spoken responses which have already been assigned numeric scores by human graders. They have also processed these responses and extracted a set of (numeric) features using systems such as `Coh-Metrix <https://soletlab.asu.edu/coh-metrix/>`_, `TextEvaluator <https://textevaluator.ets.org/TextEvaluator/>`_, `OpenSmile <https://www.audeering.com/research/opensmile/>`_, or using their own custom text/speech processing pipeline. They wish to understand how well the set of chosen features can predict the human score.

They can then run an RSMTool "experiment" to build a regression-based scoring model (using one of many available regressors) and produce a report. The report includes descriptive statistics for all their features, diagnostic information about the trained regression model, and a comprehensive evaluation of model performance on a held-out set of responses.

While they could use ``R``, ``PASW`` (``SPSS``) or other tools to perform each of the RSMTool analyses individually and compile a report himself, RSMTool does all of this work for them with just a single command. Furthermore, the analyses included into the tool highlight educational measurement criteria important to building automated scoring models. If they wish, they can conduct further exploratory analysis using their preferred tools for data analysis by using the output of RSMTool as a starting point.

RSMTool is designed to be customizable:

1. The users can choose to either run all of the default analyses or select *only* the subset applicable to their particular study by changing the appropriate settings in a configuration file.


2. RSMTool provides explicit support for adding :ref:`custom analyses <custom_notebooks>` to the report if the user has some analysis in mind that is not already provided by RSMTool. These analyses can then be automatically included in all subsequent experiments.

While training and evaluating a scoring model represents the most common use case for RSMTool, it can do a lot more for advanced users such as:

- evaluating predictions obtained from *any* ML model using :ref:`rsmeval <usage_rsmeval>`,
- generating predictions for new data from an ``rsmtool`` model using :ref:`rsmpredict <usage_rsmpredict>`,
- generating a detailed comparison between two different ``rsmtool`` models using :ref:`rsmcompare <usage_rsmcompare>`,
- generating a summary report for multiple ``rsmtool`` models using :ref:`rsmsummarize <usage_rsmsummarize>`,
- running cross-validation to estimate the performance of ``rsmtool`` models using :ref:`rsmxval <usage_rsmxval>`, and
- explaining the predictions of non-linear ``rsmtool`` models using :ref:`rsmexplain <usage_rsmexplain>`.
