.. _api:

API Documentation
-----------------
The primary method of using RSMTool is via the command-line scripts :ref:`rsmtool <usage_rsmtool>`, :ref:`rsmeval <usage_rsmeval>`, :ref:`rsmpredict <usage_rsmpredict>`, :ref:`rsmcompare <usage_rsmcompare>`, and :ref:`rsmsummarize <usage_rsmsummarize>`. However, there are certain functions in the ``rsmtool`` API that may also be useful to advanced users for use directly in their Python code. We document these functions below.

:mod:`rsmtool` Package
======================

.. autofunction:: rsmtool.run_experiment
.. autofunction:: rsmtool.run_evaluation
.. autofunction:: rsmtool.run_comparison
.. autofunction:: rsmtool.run_summary
.. autofunction:: rsmtool.compute_and_save_predictions
.. autofunction:: rsmtool.fast_predict
.. autofunction:: rsmtool.generate_explanation

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
.. autofunction:: rsmtool.convert_feature_json.convert_feature_json_file


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


From :py:mod:`~rsmtool.utils.prmse` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. _prmse_api:
.. autofunction:: rsmtool.utils.prmse.prmse_true
.. _ve_api:
.. autofunction:: rsmtool.utils.prmse.variance_of_errors

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
    :show-inheritance:

From :py:mod:`~rsmtool.transformer` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. automodule:: rsmtool.transformer
    :members:
    :undoc-members:
    :show-inheritance:

From :py:mod:`~rsmtool.utils` Module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. _generation_api:
.. autoclass:: rsmtool.utils.commandline.ConfigurationGenerator
.. automethod:: rsmtool.utils.commandline.ConfigurationGenerator.generate
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
