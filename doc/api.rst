.. _api:

API Documentation
-----------------
The primary method of using RSMTool is via the command-line scripts ``rsmtool``, ``rsmeval``, ``rsmpredict``, and ``rsmcompare``. However, there are certain functions in the ``rsmtool`` API that may also be useful to advanced users.

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
