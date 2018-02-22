.. _api:

API Documentation
-----------------
The primary method of using RSMTool is via the command-line scripts :ref:`rsmtool <usage_rsmtool>`, :ref:`rsmeval <usage_rsmeval>`, :ref:`rsmpredict <usage_rsmpredict>`, :ref:`rsmcompare <usage_rsmcompare>`, and :ref:`rsmsummarize <usage_rsmsummarize>`. However, there are certain functions in the ``rsmtool`` API that may also be useful to advanced users for use directly in their Python code. We document these functions below.

.. note::
    
     RSMTool v5.7 and older provided the API functions ``metrics_helper``, ``convert_ipynb_to_html``, and ``remove_outliers``. These functions have now been turned into static methods for different classes. If you are using these functions in your code and want to migrate to the new API, you should replace the follwing statements in your code:

    .. code-block:: python

        from rsmtool.analysis import metrics_helper
        metrics_helper(...)

        from rsmtool.report import convert_ipynb_to_html
        convert_ipynb_to_html(...)
        
        from rsmtool.preprocess import remove_outliers
        remove_outliers(...)

    with the following, respectively:

    .. code-block:: python

        from rsmtool.analyzer import Analyzer
        Analyzer.metrics_helper(...)

        from rsmtool.reporter import Reporter
        Reporter.convert_ipynb_to_html(...)
        
        from rsmtool.preprocessor import FeaturePreprocessor
        FeaturePreprocessor.remove_outliers(...)

:mod:`rsmtool` Package
======================

.. autofunction:: rsmtool.run_experiment
.. autofunction:: rsmtool.run_evaluation
.. autofunction:: rsmtool.run_comparison
.. autofunction:: rsmtool.run_summary
.. autofunction:: rsmtool.compute_and_save_predictions

From :py:mod:`~rsmtool.analyzer` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: rsmtool.analyzer
    :members:
    :undoc-members:
    :show-inheritance:

From :py:mod:`~rsmtool.comparer` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: rsmtool.comparer
    :members:
    :undoc-members:
    :show-inheritance:

From :py:mod:`~rsmtool.configuration_parser` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: rsmtool.configuration_parser
    :members:
    :undoc-members:
    :show-inheritance:

From :py:mod:`~rsmtool.container` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: rsmtool.container
    :members:
    :undoc-members:
    :show-inheritance:

From :py:mod:`~rsmtool.convert_feature_json` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: rsmtool.convert_feature_json_file

From :py:mod:`~rsmtool.modeler` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: rsmtool.modeler
    :members:
    :undoc-members:
    :show-inheritance:

From :py:mod:`~rsmtool.preprocessor` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: rsmtool.preprocessor
    :members:
    :undoc-members:
    :show-inheritance:

From :py:mod:`~rsmtool.reader` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: rsmtool.reader
    :members:
    :undoc-members:
    :show-inheritance:

From :py:mod:`~rsmtool.reporter` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: rsmtool.reporter
    :members:
    :undoc-members:
    :show-inheritance:

From :py:mod:`~rsmtool.transformer` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: rsmtool.transformer
    :members:
    :undoc-members:
    :show-inheritance:

From :py:mod:`~rsmtool.utils` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: rsmtool.utils.agreement
.. autofunction:: rsmtool.utils.partial_correlations
.. autofunction:: rsmtool.utils.get_thumbnail_as_html
.. autofunction:: rsmtool.utils.show_thumbnail
.. autofunction:: rsmtool.utils.compute_expected_scores_from_model
.. autofunction:: rsmtool.utils.parse_json_with_comments

From :py:mod:`~rsmtool.writer` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: rsmtool.writer
    :members:
    :undoc-members:
    :show-inheritance:

