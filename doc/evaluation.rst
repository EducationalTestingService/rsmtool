.. _evaluation:

Evaluation Metrics
""""""""""""""""""

This section documents the exact mathematical definitions of the primary metrics used in RSMTool for evaluating the performance of automated scoring engines. RSMTool reports also include many secondary evaluations as described in :ref:`intermediary files<rsmtool_eval_files>` and the :ref:`report sections<general_sections_rsmtool>`.
 
The following conventions are used in the formulas in this section:

:math:`N` \\-\\- total number of responses in the :ref:`evaluation set<test_file>` with numeric human scores and numeric system scores. Zero human scores are, by default, excluded from evaluations unless :ref:`exclude_zero_scores<exclude_zero_scores_rsmtool>` was set to ``false``.

:math:`M` \\-\\- system score. The primary evaluation metrics in the RSMTool report are computed for *all* six types of :ref:`scores <score_postprocessing>`. For some secondary evaluations, the user can choose between raw and scaled scores using the :ref:`use_scaled_predictions<use_scaled_predictions_rsmtool>` configuration field for RSMTool or the :ref:`scale_with<scale_with_eval>` field for RSMEval.

:math:`H` \\-\\- human score. The score values in :ref:`test_label_column<test_label_column_rsmtool>` for RSMTool or :ref:`human_score_column<human_score_column_eval>` for RSMEval.

.. _h2:

:math:`H2` \\-\\- second human score (if available). The score values in :ref:`second_human_score_column<second_human_score_column_rsmtool>`.

:math:`N_2` \\-\\- total number of responses in the evaluation set where both :math:`H` and :math:`H2` are available and are numeric and non-zero (unless :ref:`exclude_zero_scores<exclude_zero_scores_rsmtool>` was set to ``false``).

:math:`\bar{M}` \\-\\- Mean of :math:`M` ; :math:`\bar{M}`  = :math:`\displaystyle\sum_{n=1}^{N}{\frac{M_i}{N}}`

:math:`\bar{H}` \\-\\- Mean of :math:`H` ; :math:`\bar{H}` = :math:`\displaystyle\sum_{n=1}^{N}{\frac{H_i}{N}}`

:math:`\sigma_M` \\-\\- Standard deviation of :math:`M` ; :math:`\sigma_M` = :math:`\displaystyle\sqrt{\frac{\sum_{i=1}^{N}{(M_i-\bar{M})^2}}{N-1}}`

:math:`\sigma_H` \\-\\- Standard deviation of :math:`H` ; :math:`\sigma_H` = :math:`\displaystyle\sqrt{\frac{\sum_{i=1}^{N}{(H_i-\bar{H})^2}}{N-1}}` 

:math:`\sigma_{H2}` \\-\\- Standard deviation of :math:`H2` ; :math:`\sigma_{H2}` = :math:`\displaystyle\sqrt{\frac{\sum_{i=1}^{N_2}{(H2_i-\bar{H2})^2}}{N_2-1}}`


.. _observed_score_evaluation:

Accuracy Metrics (Observed score)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These metrics show how well system scores :math:`M` predict observed human scores :math:`H`. The computed metrics are available in the :ref:`intermediate file<rsmtool_eval_files>` ``eval``, with a subset of the metrics also available in the intermediate file ``eval_short``. 

.. _exact_agreement:

Percent exact agreement (rounded scores only)
+++++++++++++++++++++++++++++++++++++++++++++

Percentage responses where human and system scores match exactly. 

:math:`A = \displaystyle\sum_{i=1}^{N}\frac{w_i}{N} \times 100`

where :math:`w_i=1` if :math:`M_i = H_i` and :math:`w_i=0` if  :math:`M_i \neq H_i`

The percent exact agreement is computed using :ref:`rsmtool.utils.agreement<agreement_api>` with ``tolerance`` set to ``0``.


.. _adjacent_agreement:

Percent exact + adjacent agreement
++++++++++++++++++++++++++++++++++

Percentage responses where the absolute difference between human and system scores is ``1`` or less.

:math:`A_{adj} = \displaystyle\sum_{i=1}^{N}\frac{w_i}{N} \times 100`

where :math:`w_i=1` if :math:`|M_i-H_i| \leq 1` and :math:`w_i=0` if  :math:`|M_i-H_i| \gt 1`.

The percent exact + adjacent agreement is computed using :ref:`rsmtool.utils.agreement<agreement_api>` with ``tolerance`` set to ``1``.


.. _kappa: 

Cohen's kappa (rounded scores only)
+++++++++++++++++++++++++++++++++++

:math:`\kappa=1-\displaystyle\frac{\sum_{k=0}^{K-1}{}\sum_{j=1}^{K}{w_{jk}X_{jk}}}{\sum_{k=0}^{K-1}{}\sum_{j=1}^{K}{w_{jk}m_{jk}}}`

when :math:`k=j` then :math:`w_{jk}` = 0 and
when :math:`k \neq j` then :math:`w_{jk}` = 1

where:

- :math:`K` is the number of scale score categories (maximum observed rating - minimum observed rating + 1). Note that for :math:`\kappa` computation the values of `H` and `M` are shifted to `H-minimum_rating` and `M-minimum_rating` so that the lowest value is 0. This is done to support negative labels.

- :math:`X_{jk}` is the number times where :math:`H=j` and :math:`M=k`. 

- :math:`m_{jk}` is the percent chance agreement:

    :math:`m_{jk} = \displaystyle\sum_{k=1}^{K}{\frac{n_{k+}}{N}\frac{n_{+k}}{N}}`, where

        * :math:`n_{k+}` - total number of responses where :math:`H_i=k` 

        * :math:`n_{+k}` - total number of responses where :math:`M_i=k` 

Kappa is computed using `skll.metrics.kappa <https://skll.readthedocs.io/en/latest/api/skll.html#from-metrics-module>`_ with ``weights`` set to ``None`` and ``allow_off_by_one`` set to ``False`` (default).

.. note::
   See `this discussion <https://github.com/EducationalTestingService/skll/issues/391#issuecomment-444145567>`_ for the explanation of how the `SKLL implementation <https://skll.readthedocs.io/en/latest/api/skll.html#skll.kappa>`_  differs from the `scikit-learn implementation <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html>`_. The two implementations might produce different results if the matrix contains missing labels. For example, consider the hypothetical scenario where our predictions only contain the labels ``1``, ``2``, and ``4``. In the SKLL implementation, the missing ``3``  will be automatically added to the list of labels whereas in the scikit-learn implementation, the ``3`` would only be added if a complete list of labels was passed to the function via the optional ``labels`` keyword argument.


.. _qwk:

Quadratic weighted kappa (QWK)
++++++++++++++++++++++++++++++


Unlike :ref:`Cohen's kappa<kappa>` which is only computed for rounded scores, quadratic weighted kappa is computed for continuous scores using the following formula: 


:math:`QWK=\displaystyle\frac{2*Cov(M,H)}{Var(H)+Var(M)+(\bar{M}-\bar{H})^2}`

Note that in this case the variances and covariance are computed by dividing by ``N`` and not by ``N-1``, as in other cases.  

QWK is computed using :ref:`rsmtool.utils.quadratic_weighted_kappa<qwk_api>` with ``ddof`` set to ``0``.

See `Haberman (2019) <https://onlinelibrary.wiley.com/doi/abs/10.1002/ets2.12258>`_ for the full derivation of this formula. The discrete case is simply treated as a special case of the continuous one. 

.. note::

	In RSMTool v6.x and earlier QWK was computed using `skll.metrics.kappa <https://skll.readthedocs.io/en/latest/api/skll.html#from-metrics-module>`_ with ``weights`` set to ``"quadratic"``. Continuous scores were rounded for computation. Both formulas produce the same scores for discrete (rounded scores) but QWK values for continuous scores computed by RSMTool starting with v7.0 will be *different* from those computed by earlier versions.


.. _r: 

Pearson Correlation coefficient (r)
++++++++++++++++++++++++++++++++++++

:math:`r=\displaystyle\frac{\sum_{i=1}^{N}{(H_i-\bar{H})(M_i-\bar{M})}}{\sqrt{\sum_{i=1}^{N}{(H_i-\bar{H})^2} \sum_{i=1}^{N}{(M-\bar{M})^2}}}`

Pearson correlation coefficients is computed using `scipy.stats.pearsonr <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html>`_. 

If the variance of human or system scores is ``0`` (all scores are the same) or only one response is available, RSMTool returns ``None``.

.. note::
  In `scipy` v1.4.1 and later, the implementation uses the following formula: 

  :math:`r=\displaystyle\frac{H-\bar{H}}{\left\|H-\bar{H}\right\|_2}\cdot\frac{M-\bar{M}}{\left\|M-\bar{M}\right\|_2}`

  This implementation is more robust to very large values but is more likely to return a value slightly smaller than 1 (for example, 0.9999999999999998) for perfect correlation when `n` is small. See this `comment <https://github.com/scipy/scipy/commit/1acf46f508afa2c6d498e1001ca17e8ad98b46ef>`_ for further detail. 


.. _smd:

Standardized mean difference (SMD)
++++++++++++++++++++++++++++++++++

This metrics ensures that the distribution of system scores is centered on a point close to what is observed with human scoring.

:math:`SMD = \displaystyle\frac{\bar{M}-\bar{H}}{\sigma_H}`

SMD between system and human scores is computed using :ref:`rsmtool.utils.standardized_mean_difference<smd_api>` with the ``method`` argument set to ``"unpooled"``.

.. note::

	In RSMTool v6.x and earlier SMD was computed with the ``method`` argument set to ``"williamson"`` as described in `Williamson et al. (2012) <https://onlinelibrary.wiley.com/doi/full/10.1111/j.1745-3992.2011.00223.x>`_.  The values computed by RSMTool starting with v7.0 will be *different* from those computed by earlier versions.


.. _mse:

Mean squared error (MSE)
++++++++++++++++++++++++

The mean squared error of a machine score 𝑀 as a predictor of observed human score H:

:math:`MSE(H|M) = \displaystyle\frac{1}{N}\sum_{i=1}^{N}{(H_{i}-M_{i})^2}`

MSE is computed using `sklearn.metrics.mean_squared_error <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html>`_.

.. _r2:

Proportional reduction in mean squared error for observed score (R2)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

:math:`R2=1-\displaystyle\frac{MSE(H|M)}{\sigma_H^2}`

R2 is computed using `sklearn.metrics.r2_score <https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html>`_. If only one response is available, RSMTool returns ``None``.

.. _true_score_evaluation:

Accuracy Metrics (True score)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

According to test theory, an observed score is a combination of the true score :math:`T` and a measurement error. The true score cannot be observed, but its distribution parameters can be estimated from observed scores. Such an estimation requires that two human scores be available for *at least a* subset of responses in the evaluation set since these are necessary to estimate the measurement error component.

The true score evaluations computed by RSMTool are available in the :ref:`intermediate file<rsmtool_true_score_eval>` ``true_score_eval``. 

Proportional reduction in mean squared error for true scores (PRMSE)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

PRMSE shows how well system scores can predict true scores. This metric generally varies between 0 (random prediction) and 1 (perfect prediction), although in some cases in can take negative values (suggesting a very bad fit) or exceed 1 (suggesting that the sample size is too small to reliably estimate rater error variance). 

PRMSE for true scores is defined similarly to :ref:`PRMSE for observed scores<r2>`, but with the true score :math:`T` used instead of the observed score :math:`H`, that is, as the percentage of variance in the true scores explained by the system scores. 

:math:`PRMSE=1-\displaystyle\frac{MSE(T|M)}{\sigma_T^2}`

In the simple case where all responses have two human scores, :math:`MSE(T|M)` (**mean squared error when predicting true score with system score**) and :math:`\sigma_T^2` (**variance of true score**) are estimated from their observed score counterparts :math:`MSE(H|M)` and :math:`\sigma_H^2` as follows:

- :math:`\hat{H}` is used instead of :math:`H` to compute :math:`MSE(\hat{H}|M)` and :math:`\sigma_{\hat{H}}^2`. :math:`\hat{H}` is the average of two human scores for each response (:math:`\hat{H_i} = \frac{{H_i}+{H2_i}}{2}`). These evaluations use :math:`\hat{H}` rather than :math:`H` because the measurement errors for each rater are assumed to be random and, thus, can partially cancel out making the average :math:`\hat{H}` closer to true score :math:`T` than :math:`H` or :math:`H2`. 

- To compute estimates for true scores, the values for observed scores are adjusted for **variance of measurement errors** (:math:`\sigma_{e}^2`) in human scores defined as:

        :math:`\displaystyle\sigma_{e}^2 = \frac{1}{2 \times N_2}\sum_{i=1}^{N_2}{(H_{i} - H2_{i})^2}`

In the simple case, where **all responses are double-scored**, :math:`MSE(T|M)` is estimated as:

   :math:`MSE(T|M) = MSE(\hat{H}|M)-\displaystyle\frac{1}{2}\sigma_{e}^2`

and :math:`\sigma_T^2` is estimated as: 

   :math:`\sigma_T^2 = \sigma_{\hat{H}}^2 - \displaystyle\frac{1}{2}\sigma_{e}^2`

The PRMSE formula implemented in RSMTool is more general and can also handle the case where the number of available ratings varies across the responses (e.g.  **only a subset of responses is double-scored**). While ``rsmtool`` and ``rsmeval`` only support evaluations with two raters, the implementation of the PRMSE formula available via the :ref:`API<prmse_api>` supports cases where some of the responses have **more than two** ratings available. 

In this case, the variance of rater errors is computed as a pooled variance estimator.

We first calculate the within-subject variance of human ratings for each response, :math:`V_i`, using denominator :math:`c_i - 1`:

:math:`V_{i} = \displaystyle\frac{\sum_{j=1}^c H_{i,j} - \bar{H}_i}{c_i-1}`

where

* :math:`H_{i,j}` is the human score assigned by rater :math:`j` to response :math:`i`

* :math:`c_i` is the total number of human scores available for response :math:`i`. For double-scored responses this equals 2. 

* :math:`\bar{H}_i` is the average human rating for response :math:`i`. 

We then take a weighted average of those within-responses variances:

:math:`\sigma_{e}^2 = \frac{\sum_{i=1}^N V_{i} * (c_i-1)}{N}`

The **true score variance** :math:`\sigma_T^2` is then estimated as 

:math:`\sigma_T^2 = \displaystyle\frac{\sum_{i=1}^N c_i (\bar{H}_i - \bar{H})^2 -
(N-1) \sigma_{e}^2}{c_\cdot - \frac{\sum_{i=1}^N
c_i^2}{c_\cdot}}`

where 

* :math:`c_\cdot = \sum_{i=1}^N c_i` is the total number of observed human scores. 

* :math:`\bar{H}_i` is the average human rating for response :math:`i`. For responses with only one rating this will be the single human score `H`.

**Mean squared error** :math:`MSE(T|M)` is estimated as: 

:math:`MSE(T|M) = \displaystyle\frac{1}{c_\cdot} \left (\sum_{i=1}^N c_i (\bar{H}_i - M_i)^2  -
N\sigma_{e}^2 \right )`


The formulas are derived to ensure consistent results regardless of the number of raters and of the number of ratings availvable for each response. 

PRMSE is computed using the :ref:`rsmtool.utils.prmse_true <prmse_api>` function.


In some cases, it may be appropriate to compute variance of human errors using a different sample than the one used for main evaluations. This can be accomplished using :ref:`rsmtool.utils.variance_of_errors <ve_api>` and using an optional configuration field `rater_error_variance` in :ref:`rsmtool<rater_error_variance_rsmtool>` or :ref:`rsmtool<rater_error_variance_rsmeval>`


.. note::

	The PRMSE formula assigns higher weight to discrepancies between system scores and human scores when human score is the average of two or more human scores than when the human score is based on a single score.


Fairness
~~~~~~~~

Fairness of automated scores is an important component of RSMTool evaluations (see `Madnani et al, 2017 <https://www.aclweb.org/anthology/W17-1605/>`_).

When defining an experiment, the RSMTool user has the option of specifying which subgroups should be considered for such evaluations using :ref:`subgroups<subgroups_rsmtool>` field. These subgroups are then used in all fairness evaluations. 

All fairness evaluations are conducted on the evaluation set. The metrics are only computed for either `raw_trim` or `scale_trim` scores (see :ref:`score postprocessing<score_postprocessing>` for further details) depending on the value of :ref:`use_scaled_predictions<use_scaled_predictions_rsmtool>` in RSMTool or the value of :ref:`scale_with<scale_with_eval>` in RSMEval. 

.. _dsm:

Differences between standardized means for subgroups (DSM)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This is a standard evaluation used for evaluating subgroup differences. The metrics are available in the :ref:`intermediate files<rsmtool_eval_files>` ``eval_by_<SUBGROUP>``.

DSM is computed as follows:

1. For each group, get the *z*-score for each response :math:i, using the :math:`\bar{H}`, :math:`\bar{M}`, :math:`\sigma_H`, and :math:`\sigma_S` for system and human scores for the whole evaluation set:

        :math:`z_{H_{i}} = \displaystyle\frac{H_i - \bar{H}}{\sigma_H}`

        :math:`z_{M_{i}} = \displaystyle\frac{M_i - \bar{M}}{\sigma_M}`


2. For each response :math:i, calculate the difference between machine and human scores: :math:`z_{M_{i}} - z_{H_{i}}`

3. Calculate the mean of the difference :math:`z_{M_{i}} - z_{H_{i}}` by subgroup of interest. 

DSM is computed using :ref:`rsmtool.utils.difference_of_standardized_means<dsm_api>` with:

 ``population_y_true_observe_mn`` = :math:`\bar{H}` for the whole evaluation set

 ``population_y_pred_mn`` = :math:`\bar{M}` for the whole evaluation set

 ``population_y_true_observed_sd`` = :math:`\sigma_H` for the whole evaluation set

 ``population_y_pred_sd`` = :math:`\sigma_M` for the whole evaluation set

 .. note::

	In RSMTool v6.x and earlier, subgroup differences were computed using :ref:`standardized mean difference <SMD>` with the ``method`` argument set to ``"williamson"``. Since the differences computed in this manner were very sensitive to score distributions, RSMTool no longer uses this function to compute subgroup differences starting with v7.0.


.. _fairness_extra: 

Additional fairness evaluations
+++++++++++++++++++++++++++++++

Starting with v7.0, RSMTool includes additional fairness analyses suggested in `Loukina, Madnani, & Zechner, 2019 <https://www.aclweb.org/anthology/W19-4401/>`_. The computed metrics from these analyses are available in :ref:`intermediate files<rsmtool_fairness_eval>` ``fairness_metrics_by_<SUBGROUP>``.

These include: 

- Overall score accuracy: percentage of variance in squared error :math:`(M-H)^2` explained by subgroup membership

- Overall score difference: percentage of variance in absolute error :math:`(M-H)` explained by subgroup membership

- Conditional score difference: percentage of variance in absolute error :math:`(M-H)` explained by subgroup membership when controlling for human score

Please refer to the paper for full descriptions of these metrics. 

The fairness metrics are computed using :ref:`rsmtool.fairness_utils.get_fairness_analyses<fairness_api>`.

.. _consistency_metrics:

Human-human agreement
~~~~~~~~~~~~~~~~~~~~~~

If scores from a second human (:ref:`H2<h2>`) are available, RSMTool computes the following additional metrics for human-human agreement using only the :math:`N_2` responses, including only responses that contain numeric values for both the :math:`H` and :math:`H2` columns.

The computed metrics are available in the :ref:`intermediate file<rsmtool_consistency_files>` ``consistency``.

Percent exact agreement
+++++++++++++++++++++++

Same as :ref:`percent exact agreement for observed scores<exact_agreement>` but substituting :math:`H2` for :math:`M`.

Percent exact + ajdacent agreement
++++++++++++++++++++++++++++++++++

Same as :ref:`percent adjacent agreement for observed scores<exact_agreement>` but substituting :math:`H2` for :math:`M` and :math:`N_2` for :math:`N`.


Cohen's kappa
+++++++++++++

Same as :ref:`Cohen's kappa for observed scores<kappa>` but substituting :math:`H2` for :math:`M` and :math:`N_2` for :math:`N`.


Quadratic weighted kappa (QWK)
++++++++++++++++++++++++++++++

Same as :ref:`QWK for observed scores<qwk>` but substituting :math:`H2` for :math:`M` and :math:`N_2` for :math:`N`.


Pearson Correlation coefficient (r)
++++++++++++++++++++++++++++++++++++

Same as :ref:`r for observed scores<r>` but substituting :math:`H2` for :math:`M` and :math:`N_2` for :math:`N`.


Standardized mean difference (SMD)
++++++++++++++++++++++++++++++++++

:math:`SMD = \displaystyle\frac{\bar{H2}-\bar{H1}}{ \sqrt{\frac{\sigma_{H}^2 + \sigma_{H2}^2}{2}}}`

Unlike :ref:`SMD for human-system scores<smd>`, the denominator in this case is the "pooled" standard deviation of :math:`H1` and :math:`H2`.


Therefore, SMD between two human scores is computed using :ref:`rsmtool.utils.standardized_mean_difference<smd_api>` with the ``method`` argument set to ``"pooled"``.

.. note::

	In RSMTool v6.x and earlier, SMD was computed with the ``method`` argument set to ``"williamson"`` as described in `Williamson et al. (2012) <https://onlinelibrary.wiley.com/doi/full/10.1111/j.1745-3992.2011.00223.x>`_.  Starting with v7.0, the values computed by RSMTool will be *different* from those computed by earlier versions.

