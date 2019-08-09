.. _evaluation:

Evaluation metrics
""""""""""""""""""

This section docments the exact definition of the main metrics used at ETS for evaluating the performance of automated scoring engines. RSMTool reports include many additional evaluations documented in the descriptions of the :ref:`intermediary files<rsmtool_eval_files>` and the :ref:`report sections<general_sections_rsmtool>`.
 
The following conventions are used in the formulas in this section:

:math:`N` - total number of responses in the evaluation set with numeric human score and numeric system score. Zero human scores are by default also excluded from the evaluation unless :ref:`exclude_zero_scores<exclude_zero_scores_rsmtool>` was set to ``false``.

:math:`M` - system score. The primary evaluation analyses in the RSMTool report are conducted for *all* six types of :ref:`scores <score_postprocessing>`. For some additional evaluations, the user can pick between raw and scaled scores.

:math:`H` - human score. The score values in :ref:`test_label_column<test_label_column_rsmtool>` for RSMTool or :ref:`human_score_column<human_score_column_eval>`. 

.. _h2:
:math:`H2` - second human score (if available). The score values in :ref:`second_human_score_column<second_human_score_column_rsmtool>`

:math:`N2` - total number of responses in the evaluation set where both :math:`H` and :math:`H2` are available and are numeric and non-zero (unless :ref:`exclude_zero_scores<exclude_zero_scores_rsmtool>` was set to ``false``).

:math:`\bar{M} = \sum_{n=1}^{N}{\frac{M_i}{N}}` - mean of :math:`M`

:math:`\bar{H} = \sum_{n=1}^{N}{\frac{H_i}{N}}` - mean of :math:`H`

:math:`\sigma_M = \sqrt{\frac{\sum_{i=1}^{N}{(M_i-\bar{M})^2}}{N-1}}` - standard deviation of :math:`M`

:math:`\sigma_H = \sqrt{\frac{\sum_{i=1}^{N}{(H_i-\bar{H})^2}}{N-1}}` - standard deviation of :math:`H`

:math:`\sigma_{H2} = \sqrt{\frac{\sum_{i=1}^{N2}{(H2_i-\bar{H2})^2}}{N2-1}}` - standard deviation of :math:`H2`


.. _observed_score_evaluation:

Observed score evaluations
~~~~~~~~~~~~~~~~~~~~~~~~~~

The computed metrics are available in the :ref:`intermediate file<rsmtool_eval_files>` ``eval`` with subset of metrics available in ``eval_short``. 

.. _exact_agreement:

Percent exact agreement (rounded scores only)
+++++++++++++++++++++++++++++++++++++++++++++

Percentage responses where human and system scores match exactly. 

:math:`A = \sum_{i=1}^{N}\frac{w_i}{N} \times 100`

where :math:`w_i=1` if :math:`M_i = H_i` and :math:`w_i=0` if  :math:`M_i \neq H_i`

The percent exact agreement is computed using :ref:`rsmtool.utils.agreement<agreement_api>` with ``tolerance`` set to ``0``.


.. _adjacent_agreement:

Percent exact + ajdacent agreement
++++++++++++++++++++++++++++++++++

Percentage responses where the absolute difference between human and system scores is ``1`` or less.

:math:`A_{adj} = \sum_{i=1}^{N}\frac{w_i}{N} \times 100`

where :math:`w_i=1` if :math:`|M_i-H_i| \leq 1` and :math:`w_i=0` if  :math:`|M_i-H_i| \gt 1`.

The percent exact + adjacent agreement is computed using :ref:`rsmtool.utils.agreement<agreement_api>` with ``tolerance`` set to ``1``.


.. _kappa: 

Cohen's kappa (rounded scores only)
+++++++++++++++++++++++++++++++++++

:math:`\kappa=1-\frac{\sum_{k=0}^{K-1}{}\sum_{j=1}^{K}{w_{jk}X_{jk}}}{\sum_{k=0}^{K-1}{}\sum_{j=1}^{K}{w_{jk}m_{jk}}}`

when :math:`k=j` then :math:`w_{jk}` = 0 and
when :math:`k \neq j` then :math:`w_{jk}` = 1

where:

- :math:`K` is the number of scale score categories (maximum observed rating - minimum observed rating + 1). Note that for :math:`\kappa` computation the values of `H` and `M` are shifted to `H-minimum_rating` and `M-minimum_rating` so that the lowest value is 0. This is done to support negative labels.

- :math:`X_{jk}` is the number times where :math:`H=j` and :math:`M=k`. 

-  :math:`m_{jk}` is the percent chance agreement:

:math:`m_{jk} = \sum_{k=1}^{K}{\frac{n_{k+}}{N}\frac{n_{+k}}{N}}`

where 
* :math:`n_{k+}` - total number of responses where :math:`H_i=k` 

* :math:`n_{+k}` - total number of responses where :math:`M_i=k` 

Kappa is computed using :ref:`skll.metrics.kappa<>` with ``weights`` set to ``None`` and ``allow_off_by_one`` set to ``False`` (default).

.. _qwk:

Quadratic weighted kappa (QWK)
++++++++++++++++++++++++++++++

Quadratic weighted kappa is computed for real-value scores using the following formula: 

:math:`QWK=\frac{E[M-H]^2}{Var(H)+Var(M)+(\bar{M}-\bar{H})^2}`

QWK is computed using :ref:`rsmtool.utils.quadratic_weighted_kappa<qwk_api>` with ``ddof`` set to ``1``.

.. _note:
	In RSMTool v.6 and earlier...

Note that this formula produces different results than those computed by :ref:`skll.metrics.kappa<>` with ``weights``  set to ``quadratic`` because the latter uses formula for discreet values only. 

.. _r: 

Pearson Correlation coefficient (r)
++++++++++++++++++++++++++++++++++++

:math:`r=\frac{\sum_{i=1}^{N}{(H_i-\bar{H})(M_i-\bar{M})}}{\sqrt{\sum_{i=1}^{N}{(H_i-\bar{H})^2} \sum_{i=1}^{N}{(M-\bar{M})^2}}}`

Pearson correlation coefficients is computed using :ref:`scipy.stats.pearsonr<>`. If the variance of human or system scores is ``0`` (all scores are the same), RSMTool returns ``None``.


.. _smd:

Standardized mean difference (SMD)
++++++++++++++++++++++++++++++++++

This metrics ensures that the distribution of system scores is centered on a point close to what is observed with human scoring.

:math:`SMD = \frac{\bar{M}-\bar{H}}{\sigma_H}`

SMD between system and human scores is computed using :ref:`rsmtool.utils.standardized_mean_difference<smd_api>` with ``method`` set to ``unpooled``.

.. _note:
	In RSMTool v.6 and earlier...

.. _dsm:

Difference between standardized means for subgroups (DSM)
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++

This metrics ensures that system scores are centered on the same point as human scores for each :ref:`subgroup of interest<subgroups_rsmtool>`

DSM is computed in the following way:

1. For each group, get the *z*-score for each response, using the :math:`\bar{H}`, :math:`\bar{M}`, :math:`\sigma_H`, and :math:`\sigma_S` for system and human scores for the whole evaluation set:

:math:`z_{H_{i}} = \frac{H_i - \bar{H}}{\sigma_H}`

:math:`z_{M_{i}} = \frac{M_i - \bar{M}}{\sigma_M}`

Where i = response i

2. For each response, calculate the difference between machine and human scores: :math:`z_{M_{i}} - z_{H_{i}}`

3. Calculate the mean of the difference :math:`z_{M_{i}} - z_{H_{i}}` by subgroup of interest. 

DSM is computed using :ref:`rsmtool.utils.difference_of_standardized_means<dsm_api>` with:

 ``population_y_true_observe_mn`` = :math:`\bar{H}` for the whole evaluation set

 ``population_y_pred_mn`` = :math:`\bar{M}` for the whole evaluation set

 ``population_y_true_observed_sd`` = :math:`\sigma_H` for the whole evaluation set

 ``population_y_pred_sd`` = :math:`\sigma_M` for the whole evaluation set

 .. _note:
	In RSMTool v.6 and earlier...

.. _mse:

Mean squared error (MSE)
++++++++++++++++++++++++

The mean squared error of a machine score 𝑀 as a predictor of observed human score H:

:math:`MSE(H|M) = \frac{1}{N}\sum_{i=1}^{N}{(H_{i}-M_{i})^2}`

MSE is computed using :ref:`sklearn.metrics.mean_squared_error<>`

.. _r2:

Proportional reduction in mean squared error for observed score (R2)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

:math:`R2=1-\frac{MSE(H|M)}{\sigma_H^2}`

R2 is computed using :ref:`sklearn.metrics.r2_score<>`

.. _true_score_evaluation:

True score evaluations
~~~~~~~~~~~~~~~~~~~~~~

According to Test Theory, an observed score is a combination of true score :math:`T` and measurement error. The true score cannot be observed, but its distribution parameters can be estimated from observed scores. Such estimation requires double human scores available for at least a subset of responses in the evaluation set.

The true score evaluations computed by RSMTool are available in the :ref:`intermediate file<smtool_true_score_eval>` ``true_score_eval``. 

Proportional reduction in mean squared error for true scores (PRMSE)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

PRMSE shows how well how well system score can predict true scores. It generally varies between 0 and 1, although in some cases in can take negative values (very bad fit) or exceed 1 (very low human-human agreement). 

PRMSE for true scores is defined similar to :ref:`PRMSE for observed scored<r2>`, but with true score :math:`T` used instead of the observed score :math:`H`:

:math:`PRMSE=1-\frac{MSE(T|M)}{\sigma_T^2}`

:math:`MSE(T|M)` (mean squared error when predicting true score with system score) and :math:`\sigma_T^2` (variance of true score) are estimated from MSE and variance for observed scores with two further changes:

- :math:`\hat{H}` is used instead of :math:`H` to compute :math:`MSE(\hat{H}|M)` and :math:`\sigma_{\hat{H}}^2`. :math:`\hat{H}` is the average of two human scores for each response (:math:`\hat{H_i} = \frac{{H_i}+{H2_i}}{2}`). These evaluations use :math:`\hat{H}` rather than :math:`H` because the measurement errors for each rater are assumed to be random and thus partially cancel out making the average :math:`\hat{H}` closer to true score :math:`T` than :math:`H` or :math:`H2`. 

- To compute estimates for true scores, the values for observed scores are adjusted for **variance of measurement errors** (:math:`\sigma_{e}^2`) in human scores defined as:

:math:`\sigma_{e}^2 = \frac{1}{2 \times N2}\sum_{i=1}^{N2}{(H_{i} - H2_{i})^2}`

Thus the **mean squared error** when predicting true score with system score (MSE(T|M)) is estimated as:

:math:`MSE(T|M) = MSE(\hat{H}|M)-\frac{1}{2}\sigma_{e}^2`

The **variance of true score** (:math:`\sigma_T^2`) is estimated as: 

:math:`\sigma_T^2 = \sigma_{\hat{H}}^2 - \frac{1}{2}\sigma_{e}^2`

The PRMSE formula implemented in RSMTool allows for both all responses to be double-scored and only percentage responses to be double-scored. Note that this formula assigns higher weight to discrepancies between system scores and human score when human score is the average of two human scores than when the human score is based on a single score.

Human-human agreement
~~~~~~~~~~~~~~~~~~~~~~

If :ref:`H2<h2>` values are available, RSMTool computes the following metrics of human-human agreement using only the :math:`N2` responses with numeric values available for both :math:`H` and :math:`H2`.

The computed metrics are available in the :ref:`intermediate file<rsmtool_consistency_files>` ``consistency``.

Percent exact agreement
+++++++++++++++++++++++

Same as :ref:`percent exact agreement for observed scores<exact_agreement>` but substituting :math:`H2` for :math:`M`.

Percent exact + ajdacent agreement
++++++++++++++++++++++++++++++++++

Same as :ref:`percent adjacent agreement for observed scores<exact_agreement>` but substituting :math:`H2` for :math:`M` and :math:`N2` for :math:`N`.


Cohen's kappa
+++++++++++++

Same as :ref:`Cohen's kappa for observed scores<kappa>` but substituting :math:`H2` for :math:`M` and :math:`N2` for :math:`N`.

.. _qwk:

Quadratic weighted kappa (QWK)
++++++++++++++++++++++++++++++

Same as :ref:`QWK for observed scores<QWK>` but substituting :math:`H2` for :math:`M` and :math:`N2` for :math:`N`.

.. _r: 

Pearson Correlation coefficient (r)
++++++++++++++++++++++++++++++++++++

Same as :ref:`r for observed scores<r>` but substituting :math:`H2` for :math:`M` and :math:`N2` for :math:`N`.

.. _smd:

Standardized mean difference (SMD)
++++++++++++++++++++++++++++++++++

:math:`SMD = \frac{\bar{H2}-\bar{H1}}{ \sqrt{\frac{\sigma_{H}^2 + \sigma_{H2}^2}{2}}}`

Unlike :ref:SMD for human-system scores<smd>`, the denominator in this case is pooled standard deviation of :math:`H1` and :math:`H2`.


SMD between two human scores is computed using :ref:`rsmtool.utils.standardized_mean_difference<smd_api>` with ``method`` set to ``pooled``.

.. _note:
	In RSMTool v.6 and earlier...



