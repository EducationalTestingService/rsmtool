.. _faq:

.. |br| raw:: html

   <div style="line-height: 0; padding: 0; margin: 0"></div>


Troubleshooting and FAQ
=======================

.. rubric:: :fa:`quora`. I got the following error:
 ``No responses remaining after filtering out non-numeric feature values. No further analysis can be run.``. What happened?

.. dropdown:: :fa:`comments`  Answer
    :container: + shadow

	``rsmtool`` is designed to work with numeric features only. Non-numeric values including missing 
	values are filtered out. Some of the common reasons for the above error are:

	- The human score column or one of the features only contains non-numeric values. You can either exclude this feature or convert it to one-hot encoding. 

	- You have features with sparse representation. The solution is to replace missing values with zeros. Note that this applies even if you use ``.jsonlines`` format. If you are using ``rsmtool`` with sparse format, let us know if `this issue <https://github.com/EducationalTestingService/rsmtool/issues/480>`_ will make your life easier. 

	- You have a lot of missing feature values and none of the responses has numeric features for every single feature. Inspect :ref:`*_excluded_responses<rsmtool_excluded_responses>` to see what responses have been excluded. 

.. rubric:: :fa:`quora`. Can you pass a set of learners to the model key in the config file or do you do a separate run for each leaner you want to look at? 

.. dropdown:: :fa:`comments`  Answer
    :container: + shadow

	``rsmtool`` can't do multiple learners yet. If you want to do multiple learners, you can use RSMTool API instead of the command line.


.. rubric:: :fa:`quora`. What’s the best way to get predictions from an RSM-tool trained model on new data?

.. dropdown:: :fa:`comments`  Answer
    :container: + shadow

    Yes! We have built :ref:`rsmpredict<usage_rsmpredict>` to do just this!

.. rubric:: :fa:`quora`. Why did ``rsmtool`` change the sign of some features in ``feature.csv``? I thought ``rsmtool`` assumes a positive sign for all features? 

.. dropdown:: :fa:`comments`  Answer
    :container: + shadow

	``rsmtool`` indeed assumes default positive sign for all raw features. However, it looks like you set ``select_transformations`` to ``True`` which means that RSMTool automatically applied :ref:`transformations<select_transformations_rsmtool>` to some of the features. Some transformations such as ``inv`` (inverse transform) change the polarity of the feature.  In this case RSMTool takes this into account and changes the sign. See :ref:`note here<clever_sign_note>`.


.. rubric:: :fa:`quora`. I ran ``rsmpredict`` to generate predicted SR scores using a selected scoring model. The datasets used in the ``rsmpredict`` is the same as those in the model building. I expected the numbers of excluded/included to be the same. However, for some reason, ``rsmpredict`` generated the predicted scores for more cases than ``rsmtool``.  

.. dropdown:: :fa:`comments`  Answer
    :container: + shadow

	There are two possible reasons:
 
	- Human 0 and non-numeric scores are excluded from model evaluation but ``rsmpredict`` would still generate scores for such responses;
 
	- If you used one of th :ref:`feature selection models<automatic_feature_selection_models>`, all responses with at least one missing value for **all** features in the original feature set would be excluded from model building. ``rsmpredict`` would only use the final and likely smaller feature set. If the features with missing values were not part of the final feature set, the responses will no longer be excluded if all other values are numeric.


.. rubric:: :fa:`quora`. The relative betas did not sum to 1 in some folds. Should we be concerned about this?

.. dropdown:: :fa:`comments`  Answer
    :container: + shadow

    Pleach check if your model returned negative coefficients. Relative coefficients only make sense when all coefficients are positive. Their sum is expected to be less than 1 if there are negative coefficients. Note that if this is the case the relative cofficients will not be included into the report. 