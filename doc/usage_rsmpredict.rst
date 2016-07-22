``rsmpredict`` - Generate new predictions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RSMTool provides the ``rsmpredict`` command-line utility to generate predictions for new data using a model already trained using the ``rsmtool`` utility. This can be useful when processing a new set of responses to the same task without needing to retrain the model.

``rsmpredict`` pre-processes the feature values according to user specifications before using them to generate the predicted scores. The generated scores are post-processed in the same manner as they are in ``rsmtool`` output.


Input
"""""
``rsmpredict`` requires two arguments to generate predictions: the path to a configuration file and the path to the output file where the generated predictions are saved in ``.csv`` format.

If you also want to save the pre-processed feature values,``rsmpredict`` can take a third optional argument ``--features`` to specify the path to a ``.csv`` file to save these values.

.. include:: config_rsmpredict.rst

Output
""""""

``rsmpredict`` produces a ``.csv`` file with predictions for all responses in new data set, and, optionally, a ``.csv`` file with pre-processed feature values.

