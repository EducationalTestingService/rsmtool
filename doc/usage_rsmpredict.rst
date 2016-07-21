RSMPredict
^^^^^^^

``rsmpredict`` generates new predictions based on an existing model including feature pre-processing and score post-processing as specified by the user. Its most common use is to generate predictions for a new data set using previously trained model. The model must be trained by using a recent version of ``rsmtool``.

.. note::
``rsmpredict`` will generate predictions for all responses in the supplied feature matrix which have numeric feature values for features included into the model. It does not require human labels. If you have the human scores and want to evaluate the new predictions, run `rsmeval <usage_rsmeval>` after you generated the predictions. Note that `rsmeval` will only do the evaluations for the responses which received numeric non-zero human score. 

Input
"""""

``rsmpredict`` requires two arguments to generate predictions: the path to the configuration file and the path to the output file where ``rsmpredict`` will save the new predictions.  If you also want to save the pre-processed features,``rsmpredict`` can also take a third optional arguments ``-- features`` which should specify the optional path to file for saving the pre-processed feature values.

.. include:: config_rsmpredict.rst

Output
""""""

``rsmpredict`` produces a ``.csv`` file with predictions for all responses in new data optionally a .csv file with pre-processed feature values requested by using ``--feature ``flag.

