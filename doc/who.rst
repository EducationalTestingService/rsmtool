Who is RSMTool for?
===================

We expect the primary users of RSMTool to be researchers working on developing new automated scoring engines or on improving existing ones. Here's the most common use case.

A researcher already *has* a set of responses such as essays or recorded spoken responses which have already been assigned numeric scores by human graders. She has also processed these responses and extracted a set of (numeric) features using systems such as Coh-Metrix, TextEvaluator, OpenSmile, or using her own custom text/speech processing pipeline.

She can then use RSMTool to build a regression-based scoring model (using one of many available regressors) for predicting the scores using the extracted features. As part of this exercise, RSMTool also provides her with: descriptive statistics for all her features, diagnostic information about the trained regression model, and a comprehensive evaluation of model performance on a held-out set of responses.

While she could use ``R``, ``SPSS`` or other tools to perform each of the RSMTool analyses individually and compile a report herself, RSMTool does all of this work with just a single command. Furthermore, the analyses included into the tool highlight educational measurement criteria important to building automated scoring models. She can then use the RSMTool outputs for further exploratory analysis of her model, if she so wishes.

Since RSMTool is designed to be customizable, the user can choose to either run all of the default analyses or select *only* the subset applicable to her particular study by changing the appropriate settings in a configuration file. Since the report is based on IPython notebooks, it can be easily customized. In addition, RSMTool provides explicit support for adding custom notebooks to the report, if the user has some analysis in mind that is not already provided via RSMTool.

While training and evaluating a scoring model represents the default use case for RSMTool, it can do a lot more for :doc:`advanced users <advanced_usage>`.


