.. _faq:

.. |br| raw:: html

   <div style="line-height: 0; padding: 0; margin: 0"></div>


Troubleshooting and FAQ
=======================

.. rubric:: :fa:`quora`. Why did RSMTool change the sign of some features in ``feature.csv``? I thought RSMTool assumes a positive sign for all features? 

.. dropdown:: :fa:`comments`  Answer
    :container: + shadow

	RSMTool indeed assumes default positive sign for all raw features. However, if you set ``select_transformations`` to True, some transformations change the polarity of the feature in which case RSMTool takes this into account and changes the sign. See note here: https://rsmtool.readthedocs.io/en/stable/usage_rsmtool.html?highlight=sign#signs In your example, the features with negative sign have all been assigned inverse transform (or addOneInv) which changes the feature polarity and therefore the sign was flipped to -1 to accommodate this. 



.. rubric:: :fa:`quora`. I ran RSMPREDICT to generate predicted SR scores using a selected scoring model. The datasets used in the RSMPREDICT is the same as those in the model building. As I assumed that RSMPREDICT would use the betas from the specified scoring model to generate SR scores, I expected the numbers of excluded/included to be the same. However, for some reason, RSMPREDICT generated the predicted scores for more cases as you can see in the table below.  

.. dropdown:: :fa:`comments`  Answer
    :container: + shadow

	Two possible reasons:
 
	(1) Human 0 and TD are excluded from model evaluation but RSMPredict would still generate scores for such responses;
 
	(2) If you used feature selection model, all responses with at least one missing value for all features in the original feature set would be excluded from model building. RSMPredict would use the final and likely smaller feature set. If the features with missing values were not part of the final feature set, the responses will no longer be excluded if all other values are numeric.

.. rubric:: :fa:`quora`.  This model produced negative coefficients for some features. The values were not large

.. dropdown:: :fa:`comments`  Answer
    :container: + shadow

	Small negative coefficients in "LassoFixedLambdaThenLR” are not unexpected: I think you already saw the note in the documentation for this model: “Note that while the original Lasso model is constrained to positive coefficients only, small negative coefficients may appear when the coefficients are re-estimated using OLS regression.” To avoid this you can use LassoFixedLambdaThenNNLR. This model should not result in negative coefficients.


.. rubric:: :fa:`quora`. The relative betas from LASSO did not sum to 1 in some folds. Should we be concerned about this?

.. dropdown:: :fa:`comments`  Answer
    :container: + shadow

    Relative coefficients only make sense when all coefficients are positive. Their sum is expected to be less than 1 if there are negative coefficients (as is the case here). This is why if you look at the report you will not see relative coefficient plot there.  Let me think where we could add this in documentation.