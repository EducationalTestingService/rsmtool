Who is RSMTool for?
===================

We expect the primary users of RSMTool to be researchers working on developing new automated scoring engines or on improving existing ones. Here's the most common scenario.

A researcher already *has* a set of responses such as essays or recorded spoken responses which have already been assigned numeric scores by human graders. She has also processed these responses and extracted a set of (numeric) features using systems such as ``Coh-Metrix``, ``TextEvaluator``, ``OpenSmile``, or using her own custom text/speech processing pipeline. She wishes to understand how well the set of chosen features can predict the human score.

She can then run an RSMTool "experiment" to build a regression-based scoring model (using one of many available regressors) and produce a report. The report includes descriptive statistics for all her features, diagnostic information about the trained regression model, and a comprehensive evaluation of model performance on a held-out set of responses.

While she could use ``R``, ``PASW`` (``SPSS``) or other tools to perform each of the RSMTool analyses individually and compile a report herself, RSMTool does all of this work for her with just a single command. Furthermore, the analyses included into the tool highlight educational measurement criteria important to building automated scoring models. If she wishes, she can conduct further exploratory analysis using her preferred tools for data analysis by using the output of RSMTool as a starting point.

RSMTool is designed to be customizable:

1. The user can choose to either run all of the default analyses or select *only* the subset applicable to her particular study by changing the appropriate settings in a configuration file.

2. RSMTool provides explicit support for adding custom analyses to the report if the user has some analysis in mind that is not already provided by RSMTool. These analyses can then be automatically included in all subsequent experiments.

While training and evaluating a scoring model represents the most common use case for RSMTool, it can do a lot more for :doc:`advanced users <advanced_usage>` such as evaluating predictions obtained using an external scoring engine, generating predictions for new data, and comparing two different scoring models.




