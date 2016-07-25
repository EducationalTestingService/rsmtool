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

From :py:mod:`~rsmtool.preprocess` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: rsmtool.trim
.. autofunction:: rsmtool.remove_outliers
