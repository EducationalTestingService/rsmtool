.. _feature_selection:

Selecting Feature Columns
-------------------------

By default, ``rsmtool`` will use all columns included in the training and evaluation ``.csv`` files as features. The only exception are the columns specified in the :ref:`rsmtool configuration file <flag_column_rsmtool>` as containing ``id``, ``train_label``, ``subgroups`` or other information. However, it is possible for you to define a subset of columns to be used as features. You can then either use all these features in the final model or use one of the built-in models that perform :ref:`automatic feature selection <automatic_feature_selection_models>`.

.. note::

    For all feature selection methods ``rsmtool`` will pre-process the data to remove all responses with non-numeric feature values for any of the features before trying to train a model. If your data set includes a column containing character (string) values and you do not specify this column as an ``id`` column, ``candidate`` column or ``subgroup`` column, ``rsmtool`` will assume that this column contains a feature and will filter out all responses in the data set. 


.. note::

    For all feature selection methods, the final set of features will be saved in the ``feature`` folder in the experiment output directory.

.. _manual_feature_selection:

Defining Feature Columns using ``.json`` file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To manually select a subset of columns to be used as features, you can provide a ``.json`` file specifying information about what columns should be included as features in the final scoring model. For additional flexibility, the same file also allows you to describe transformations to be applied to the raw features before being included in the model.

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
The name of the feature. This must match the feature name in the training and evaluation ``.csv`` files exactly, including capitalization. Feature names should *not* contain hyphens. The following names are reserved and should not be used as feature names: ``spkitemid``, ``spkitemlab``, ``itemType``, ``r1``, ``r2``, ``score``, ``sc``, ``sc1``, and ``adj``. In addition, any column names provided as values for  ``id_column``, ``train_label_column``, ``test_label_column``, ``length_column``, ``candidate_column``, and ``subgroups`` can also not be used as feature names.

transform
"""""""""
A transformation that should be applied to the feature values before using it in the model. Possible values are:

    * ``raw``: no transformation, use original feature value
    * ``org``: same as raw
    * ``inv``: 1/x
    * ``sqrt``: square root
    * ``addOneInv``: 1/(x+1)
    * ``addOneLn``: ln(x+1)

Note that ``rsmtool`` will raise an exception if the values in the data do not allow the supplied transformation (for example, if ``inv`` is applied to a feature which has 0 values). If you really want to use the tranformation, you must pre-process your training and evaluation ``.csv`` files to remove the problematic cases.

sign
""""

After transformation, each feature value will be multiplied by this number. This field is usually set to ``1`` or ``-1`` depending on the expected sign of the correlation between transformed feature and human score to ensure that all features in the final models have positive correlation with the score.

When determining the sign, you should take into account the correlation between the original feature and the score as well as any applied transformations.  For example, if you use feature which has a negative correlation with the human score and apply ``sqrt`` transformation, ``sign`` should be set to ``-1``. However, if you use the same feature but apply the ``inv`` transformation, ``sign`` should now be set to ``1``.

To ensure that this is working as expected, you can check the sign of correlations for both raw and processed features in the final report.

.. _subset_feature_selection:

Defining Feature Columns using ``.csv`` file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For more advanced users, ``rsmtool`` offers the ability to define feature subsets in a ``.csv`` file and then select groups of columns to be used as features by simply specifying the pre-defined subset to use.

This function can be useful if the software you use for feature extraction generates many features, you want to build scoring models to score different types of questions, and for each type of question you only want to use a subset of features. In this case you would need to either pre-process your data to remove the columns with unused features or define the list of columns to use as features. While you can generate a separate :ref:`json file <manual_feature_selection>` to list features you want to use for each type of questions, this can be a cumbersome process if the subsets are large. 

Instead you can define feature subsets by providing a master ``.csv`` file which lists *all* feature names that you might want to use under a column named ``Feature``. Then each subset is an additional column with the value of either ``0`` (denoting that the feature does *not* belong to the subset named by that column) or ``1`` (denoting that the feature does belong to the subset). 

This ``.csv`` file can be provided to ``rsmtool`` using the :ref:`feature_subset_file <feature_subset_file>` field in the configuration file. Then, to select a particular pre-defined subset of features, you can simply set the :ref:`feature_subset  <feature_subset>` field in the configuration file to the name of the subset that you wish to use.

Unlike :ref:`json file <manual_feature_selection>`, ``.csv`` file does not contain information about transformation and sign for each feature. 

RSMTool can automatically select transformation for each feature by applying all possible transforms and identifying the one which gives the highest correlation with the human score. To use this functionality set the :ref:`select_transformations <select_transformations>` field in the configuration file to ``true``. 

Most guidelines for building scoring models require that all coefficients in the model are positive and that all features have a positive correlation with human score. ``rsmtool`` can automatically flip the sign for any pre-defined feature subset. To use this functionality, the feature subset ``.csv`` file should provide the expected correlation sign between each feature and human score under a column called ``sign_<SUBSET>`` where ``<SUBSET>`` is the name of the feature subset. Then, to tell ``rsmtool`` to flip the the sign, you need to set the :ref:`sign <sign>` field in the configuration file to ``<SUBSET>``. 

.. note::

    If :ref:`select_transformations <select_transformations>` is set to ``true``, ``rsmtool`` will take into account the transformation applied to the features. Thus if the expected correlation sign for a given feature is negative, ``rsmtool`` will multiply the feature values by ``-1`` if no transformation is applied. However, no such multiplication will be applied if the feature is transformed using ``inverse`` tranform, which already changes the polarity of the feature. 



Example
"""""""

It's best to illustrate subset-based selection with an example. Let's say that we have a feature subset definition file called ``subset.csv``:

.. code-block:: text

    Feature,A,B,sign_A
    feature1,0,1,+
    feature2,1,1,-
    feature3,1,1,+

Then, in order to use the subset "A" of features in an experiment (``feature2`` and `feature3` only) with the sign of ``feature3`` flipped appropriately (multiplied by -1) to ensure positive correlations with score and positive model coefficients, we need to set the following three fields in our experiment configuration file:

.. code-block:: javascript

    {
        ...
        "feature_subset_file": "subset.csv",
        "feature_subset": "A",
        "sign": "A"
        ...
    }

.. note::

    While for most users different ``sign`` values will correspond to different ``subsets``, this is not a requirement. You can have ``sign`` set to ``A`` while setting ``subset`` to ``B``.