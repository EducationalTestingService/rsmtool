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

From :py:mod:`~rsmtool:create_features.py` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: rsmtool.generate_default_specs
.. autofunction:: rsmtool.find_feature_transformation

From :py:mod:`~rsmtool:input.py` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: rsmtool.parse_json_with_comments

From :py:mod:`~rsmtool.report` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: rsmtool.convert_ipynb_to_html
