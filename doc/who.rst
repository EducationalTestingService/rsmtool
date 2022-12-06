.. _who_rsmtool:

Who is RSMTool for?
===================

We expect the primary users of RSMTool to be researchers working on developing new automated scoring engines or on improving existing ones. Here's the most common scenario.

A group of researchers already *has* a set of responses such as essays or recorded spoken responses which have already been assigned numeric scores by human graders. They have also processed these responses and extracted a set of (numeric) features using systems such as `Coh-Metrix <http://cohmetrix.com/>`_, `TextEvaluator <https://textevaluator.ets.org/TextEvaluator/>`_, `OpenSmile <https://www.audeering.com/research/opensmile/>`_, or using their own custom text/speech processing pipeline. They wish to understand how well the set of chosen features can predict the human score.

They can then run an RSMTool "experiment" to build a regression-based scoring model (using one of many available regressors) and produce a report. The report includes descriptive statistics for all their features, diagnostic information about the trained regression model, and a comprehensive evaluation of model performance on a held-out set of responses.

While they could use ``R``, ``PASW`` (``SPSS``) or other tools to perform each of the RSMTool analyses individually and compile a report himself, RSMTool does all of this work for them with just a single command. Furthermore, the analyses included into the tool highlight educational measurement criteria important to building automated scoring models. If they wish, they can conduct further exploratory analysis using their preferred tools for data analysis by using the output of RSMTool as a starting point.

RSMTool is designed to be customizable:

1. The users can choose to either run all of the default analyses or select *only* the subset applicable to their particular study by changing the appropriate settings in a configuration file.


2. RSMTool provides explicit support for adding :ref:`custom analyses <custom_notebooks>` to the report if the user has some analysis in mind that is not already provided by RSMTool. These analyses can then be automatically included in all subsequent experiments.

While training and evaluating a scoring model represents the most common use case for RSMTool, it can do a lot more for advanced users such as :ref:`evaluating predictions obtained using an external scoring engine <usage_rsmeval>`, :ref:`generating predictions for new data <usage_rsmpredict>`, :ref:`generating a detailed comparison between two different scoring models <usage_rsmcompare>`, :ref:`generating a summary report for multiple scoring models <usage_rsmsummarize>`, and :ref:`using cross-validation to better estimate the performance of scoring models <usage_rsmxval>`.
