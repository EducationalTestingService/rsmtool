.. _feature_selection:

Selecting Features
------------------

By default, ``rsmtool`` will use all of the features provided in the training and evaluation ``.csv`` files. However, it is possible for you to either manually choose a subset of features or to have ``rsmtool`` perform automatic feature selection.

.. note::

    For all feature selection methods, the final set of features will be saved in the ``feature`` folder in the experiment output directory.

.. _manual_feature_selection:

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

Automatic Feature selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^
No feature file is necessary for models with automatic feature selection. You simply need to pick one of the built-in models that performs :ref:`automatic feature selection <automatic_feature_selection_models>`.

.. _subset_feature_selection:

Subset-based Feature Selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For more advanced users, ``rsmtool`` offers the ability to define feature subsets and then select groups of features by simply specifying the pre-defined subset to use.

To define feature subsets, you must provide a master ``.csv`` file which lists *all* features that should be used for feature selection. Each feature name should be listed under a column named ``Feature``. Then each subset is an additional column with the value of either ``0`` (denoting that the feature does *not* belong to the subset named by that column) or ``1`` (denoting that the feature does belong to the subset).

This ``.csv`` file can be provided to ``rsmtool`` using the :ref:`feature_subset_file <feature_subset_file>` field in the configuration file. Then, to select a particular pre-defined subset of features, you can simply set the :ref:`feature_subset  <feature_subset>` field in the configuration file to the name of the subset that you wish to use.

Most guidelines for building scoring models require that all coefficients in the model are positive and that all features have a positive correlation with human score. ``rsmtool`` can automatically flip the signs for any pre-defined feature subset. To use this functionality, the feature subset ``.csv`` file should provide the expected correlation sign between each feature and human score under a column called ``sign_<SUBSET>`` where ``<SUBSET>`` is the name of the feature subset. Then, to tell ``rsmtool`` to flip the signs, you need to set the :ref:`sign <sign>` field in the configuration file to ``<SUBSET>``.

Example
"""""""

It's best to illustrate subset-based selection with an example. Let's say that that we have a feature subset definition file called ``subset.csv``:

.. code-block:: text

    Feature,A,sign_A
    feature1,0,+
    feature2,1,-
    feature3,1,+

Then, in order to use the subset "A" of features in an experiment with the sign of ``feature3`` flipped appropriately (multiplied by -1) to ensure positive correlations with score and positive model coefficients, we need to set the following three fields in our experiment configuration file:

.. code-block:: javascript

    {
        ...
        "feature_subset_file": "subset.csv",
        "feature_subset": "A",
        "sign": "A"
        ...
    }
