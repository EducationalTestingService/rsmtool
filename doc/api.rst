.. _api:

API Documentation
-----------------
The primary method of using RSMTool is via the command-line scripts :ref:`rsmtool <usage_rsmtool>`, :ref:`rsmeval <usage_rsmeval>`, :ref:`rsmpredict <usage_rsmpredict>`, and :ref:`rsmcompare <usage_rsmcompare>`. However, there are certain functions in the ``rsmtool`` API that may also be useful to advanced users for use directly in their Python code. We document these functions below.

:mod:`rsmtool` Package
======================

From :py:mod:`~rsmtool.analysis` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: rsmtool.compute_basic_descriptives
.. autofunction:: rsmtool.compute_percentiles
.. autofunction:: rsmtool.compute_outliers
.. autofunction:: rsmtool.compute_pca
.. autofunction:: rsmtool.correlation_helper
.. autofunction:: rsmtool.metrics_helper

From :py:mod:`~rsmtool.preprocess` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: rsmtool.trim
.. autofunction:: rsmtool.remove_outliers
.. autofunction:: rsmtool.filter_on_column
.. autofunction:: rsmtool.transform_feature
.. autofunction:: rsmtool.preprocess_feature

From :py:mod:`~rsmtool.create_features` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: rsmtool.generate_default_specs
.. autofunction:: rsmtool.find_feature_transformation

From :py:mod:`~rsmtool.input` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: rsmtool.parse_json_with_comments

From :py:mod:`~rsmtool.model` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: rsmtool.model_fit_to_dataframe
.. autofunction:: rsmtool.ols_coefficients_to_dataframe
.. autofunction:: rsmtool.skll_learner_params_to_dataframe
.. autofunction:: rsmtool.train_builtin_model


From :py:mod:`~rsmtool.predict` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: rsmtool.predict_with_model


From :py:mod:`~rsmtool.report` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: rsmtool.convert_ipynb_to_html
.. autofunction:: rsmtool.merge_notebooks


From :py:mod:`~rsmtool.utils` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: rsmtool.partial_correlations
.. autofunction:: rsmtool.agreement
.. autofunction:: rsmtool.write_experiment_output
.. autofunction:: rsmtool.write_feature_json

