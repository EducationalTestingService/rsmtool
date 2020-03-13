.. _api:

API Documentation
-----------------
The primary method of using RSMTool is via the command-line scripts :ref:`rsmtool <usage_rsmtool>`, :ref:`rsmeval <usage_rsmeval>`, :ref:`rsmpredict <usage_rsmpredict>`, :ref:`rsmcompare <usage_rsmcompare>`, and :ref:`rsmsummarize <usage_rsmsummarize>`. However, there are certain functions in the ``rsmtool`` API that may also be useful to advanced users for use directly in their Python code. We document these functions below.

.. note::
    
    RSMTool v5.7 and older provided the API functions ``metrics_helper``, ``convert_ipynb_to_html``, and ``remove_outliers``. These functions have now been turned into static methods for different classes. 

    In addition, with RSMTool v8.0 onwards, the functions ``agreement``, ``difference_of_standardized_means``, ``get_thumbnail_as_html``,  ``parse_json_with_comments``, ``partial_correlations``,  ``quadratic_weighted_kappa``, ``show_thumbnail``,  and ``standardized_mean_difference`` that ``utils.py`` had previously provided have been moved to new locations.

    If you are using the above functions in your code and want to migrate to the new API, you should replace the following statements in your code:

    .. code-block:: python

     from rsmtool.analysis import metrics_helper
     metrics_helper(...)

     from rsmtool.report import convert_ipynb_to_html
     convert_ipynb_to_html(...)
        
     from rsmtool.preprocess import remove_outliers
     remove_outliers(...)

     from rsmtool.utils import agreement
     agreement(...)
        
     from rsmtool.utils import difference_of_standardized_means
     difference_of_standardized_means(...)
        
     from rsmtool.utils import partial_correlations
     partial_correlations(...)
        
     from rsmtool.utils import quadratic_weighted_kappa
     quadratic_weighted_kappa(...)
        
     from rsmtool.utils import standardized_mean_difference
     standardized_mean_difference(...)
        
     from rsmtool.utils import parse_json_with_comments
     parse_json_with_comments(...)

     from rsmtool.utils import get_thumbnail_as_html
     get_thumbnail_as_html(...)

     from rsmtool.utils import show_thumbnail
     show_thumbnail(...)

    with the following, respectively:

    .. code-block:: python

     from rsmtool.analyzer import Analyzer
     Analyzer.metrics_helper(...)

     from rsmtool.reporter import Reporter
     Reporter.convert_ipynb_to_html(...)
    
     from rsmtool.preprocessor import FeaturePreprocessor
     FeaturePreprocessor.remove_outliers(...)

     from rsmtool.utils.metrics import agreement
     agreement(...)
    
     from rsmtool.utils.metrics import difference_of_standardized_means
     difference_of_standardized_means(...)
    
     from rsmtool.utils.metrics import partial_correlations
     partial_correlations(...)
    
     from rsmtool.utils.metrics import quadratic_weighted_kappa
     quadratic_weighted_kappa(...)
    
     from rsmtool.utils.metrics import standardized_mean_difference
     standardized_mean_difference(...)
    
     from rsmtool.utils.files import parse_json_with_comments
     parse_json_with_comments(...)

     from rsmtool.utils.notebook import get_thumbnail_as_html
     get_thumbnail_as_html(...)

     from rsmtool.utils.notebook import show_thumbnail
     show_thumbnail(...)


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
    :exclude-members: deprecated_positional_argument

From :py:mod:`~rsmtool.container` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: rsmtool.container
    :members:
    :undoc-members:
    :show-inheritance:

From :py:mod:`~rsmtool.convert_feature_json` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autofunction:: rsmtool.convert_feature_json_file


From :py:mod:`~rsmtool.fairness_utils` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. _fairness_api:
.. autofunction:: rsmtool.fairness_utils.get_fairness_analyses

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


From :py:mod:`~rsmtool.prmse_utils` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. _prmse_api:
.. autofunction:: rsmtool.prmse_utils.compute_prmse

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

.. autofunction:: rsmtool.utils.commandline.generate_configuration
.. _agreement_api:
.. autofunction:: rsmtool.utils.metrics.agreement
.. _dsm_api:
.. autofunction:: rsmtool.utils.metrics.difference_of_standardized_means
.. autofunction:: rsmtool.utils.metrics.partial_correlations
.. _qwk_api:
.. autofunction:: rsmtool.utils.metrics.quadratic_weighted_kappa
.. _smd_api:
.. autofunction:: rsmtool.utils.metrics.standardized_mean_difference
.. autofunction:: rsmtool.utils.metrics.compute_expected_scores_from_model
.. autofunction:: rsmtool.utils.notebook.get_thumbnail_as_html
.. autofunction:: rsmtool.utils.notebook.show_thumbnail
.. autofunction:: rsmtool.utils.files.parse_json_with_comments

From :py:mod:`~rsmtool.writer` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: rsmtool.writer
    :members:
    :undoc-members:
    :show-inheritance:
