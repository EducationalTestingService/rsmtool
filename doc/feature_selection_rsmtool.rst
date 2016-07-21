.. _feature_selection:

Selecting Features
------------------

By default, ``rsmtool`` will use all of the features provided in the training and evaluation ``.csv`` files. However, it is possible for you to either manually choose a subset of features or to have ``rsmtool`` perform automatic feature selection.

Manual Feature selection
^^^^^^^^^^^^^^^^^^^^^^^^
To manually select a subset of features, you must provide a ``.json`` file specifying information about what features should be included in the final scoring model. For additional flexibility, the same file also allows you to describe transformations to be applied to the raw features before being included in the model.

Here's an example of what this file looks like.


.. code-block:: javascript

    {
        "features": [{
                         "feature": "feature1",
                         "transform": "raw",
                         "sign": 1,
                     },
                     {
                         "feature": "feature2",
                         "transform": "inv",
                         "sign": -1,
                     },
                     ...
                    ]
    }


There are three required fields.

feature
"""""""
The name of the feature. This must match the feature name in the training and evaluation ``.csv`` files exactly, including capitalization. Feature names should *not* contain hyphens. The following names are reserved and should not be used as feature names: ``spkitemid``, ``spkitemlab``, ``itemType``, ``r1``, ``r2``, ``score``, ``sc``, ``sc1``, and ``adj``.

transform
"""""""""
A transformation that should be applied to the feature values before using it in the model. Possible values are:

    .. hlist::

    * ``raw`` - no transformation, use original feature value
    * ``org`` - same as raw
    * ``inv`` - 1/x
    * ``sqrt`` - square root
    * ``addOneInv`` - 1/(x+1)
    * ``addOneLn`` - ln(x+1)

Note that ``rsmtool`` will return an error if the values in the data do not allow the supplied transformation (for example, if ``inv`` is applied to a feature which has 0 values). If you really want to use the tranformation, you must pre-process your training and evaluation ``.csv`` files to remove the problematic cases.

sign
""""

After transformation, each feature value will be multiplied by this number. This field is usually set to ``1`` or ``-1`` depending on the expected sign of the correlation between transformed feature and human score to ensure that all features in the final models have positive correlation with the score.

When determining the sign, you should take into account the correlation between the original feature and the score as well as any applied transformations.  For example, if you use feature which has a negative correlation with the human score and apply ``sqrt`` transformation, ``sign`` should be set to ``-1``. However, if you use the same feature but apply the ``inv`` transformation, ``sign`` should now be set to ``1``.

To ensure that this is working as expected, you can check the sign of correlations for both raw and processed features in the final report.

Automatic Feature selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^

No feature file is necessary for models with automatic feature selection. You simply need to pick one of the built-in models that performs :ref:`automatic feature selection <automatic_feature_selection_models>`.
