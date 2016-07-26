.. _column_selection_rsmtool:

Selecting Feature Columns
-------------------------

By default, ``rsmtool`` will use all columns included in the training and evaluation ``.csv`` files as features. The only exception are any columns explicitly identified in the configuration file as containing non-feature information (e.g., :ref:`id_column <id_column_rsmtool>`, :ref:`train_label_column <train_label_column_rsmtool>`, :ref:`test_label_column <test_label_column_rsmtool>`, etc.)

However, there are certain scenarios in which it is useful to choose specific columns in the data to be used as features. For example, let's say that you have a large number of very different features and you want to use a different subset of features to score different types of questions on a test. In this case, the ability to easily choose the desired features for any ``rsmtool`` experiment becomes quite important. The alternative of manually pre-processing the data to remove the features you don't need is quite cumbersome.

There are two ways to select specific columns in the data as features:

    1. **Fine-grained column selection**: In this method, you manually create a list of the columns that you wish to use as features for an ``rsmtool`` experiment. See :ref:`fine-grained selection <json_column_selection>` for more details.

    2. **Subset-based column selection**: In this method, you can pre-define subsets of features and then select entire subsets at a time for any ``rsmtool`` experiment. See :ref:`subset-based selection <subset_column_selection>` for more details.

While fine-grained column selection is better for a single experiment, subset-based selection is more convenient when you need to run several experiments with somewhat different subsets of features.

.. warning::

    ``rsmtool`` will filter the training and evaluation data to remove any responses with non-numeric values in any of the feature columns before training the model. If your data includes a column containing string values and you do *not* use any of these methods of feature selection *nor* specify this column as the ``id_column`` or the ``candidate_column``  or a ``subgroup`` column, ``rsmtool`` will filter out *all* the responses in the data.


.. _json_column_selection:

Fine-grained column selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To manually select columns to be used as features, you can provide a ``.json`` file specifying the names of the columns that should be used for scoring model building. For additional flexibility, the same file also allows you to describe transformations to be applied to the the values in these feature columns before being used in the model.

.. _example_feature_json:

Here's an example of what such a file might look like.

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
The exact name of the column in the training and evaluation ``.csv`` files, including capitalization. Column names cannot contain hyphens. The following strings are reserved and cannot not be used as feature column names: ``spkitemid``, ``spkitemlab``, ``itemType``, ``r1``, ``r2``, ``score``, ``sc``, ``sc1``, and ``adj``. In addition, any column names provided as values for  ``id_column``, ``train_label_column``, ``test_label_column``, ``length_column``, ``candidate_column``, and ``subgroups`` may also not be used as feature column names.

transform
"""""""""
A transformation that should be applied to the column values before using it in the model. Possible values are:

    * ``raw``: no transformation, use original value
    * ``org``: same as raw
    * ``inv``: 1/x
    * ``sqrt``: square root
    * ``addOneInv``: 1/(x+1)
    * ``addOneLn``: ln(x+1)

Note that ``rsmtool`` will raise an exception if the values in the data do not allow the supplied transformation (for example, if ``inv`` is applied to a column which has 0 values). If you really want to use the tranformation, you must pre-process your training and evaluation ``.csv`` files to remove the problematic cases.

sign
""""
After transformation, the column values will be multiplied by this number. This field is usually set to ``1`` or ``-1`` depending on the expected sign of the correlation between transformed feature and human score to ensure that all features in the final models have positive correlation with the score.

When determining the sign, you should take into account the correlation between the original feature and the score as well as any applied transformations.  For example, if you use feature which has a negative correlation with the human score and apply ``sqrt`` transformation, ``sign`` should be set to ``-1``. However, if you use the same feature but apply the ``inv`` transformation, ``sign`` should now be set to ``1``.

To ensure that this is working as expected, you can check the sign of correlations for both raw and processed features in the final report.

.. note::

        You can use the fine-grained method of column selection in combination with a :ref:`model with automatic feature selection <automatic_feature_selection_models>`. In this case, if not all features are used in the final model, ``rsmtool`` will create a new ``.json`` file in the ``feature`` folder in the experiment output directory. This file will only contain information about the features included in the final model.

.. _subset_column_selection:

Subset-based column selection
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For more advanced users, ``rsmtool`` offers the ability to assign columns to named subsets in a ``.csv`` file and then select a set of columns by simply specifying the name of that pre-defined subset.

If you want to run multiple ``rsmtool`` experiments, each choosing from a large number of features, generating a separate :ref:`json file <json_column_selection>` for each experiment listing columns to use can quickly become tedious.

Instead you can define feature subsets by providing a subset definition ``.csv`` file which lists *all* feature names under a column named ``Feature``. Each subset is an additional column with a value of either ``0`` (denoting that the feature does *not* belong to the subset named by that column) or ``1`` (denoting that the feature does belong to the subset named by that column).

Here's an example of a subset definition file, say ``subset.csv``.

.. code-block:: text

    Feature,A,B
    feature1,0,1
    feature2,1,1
    feature3,1,0

In this example, ``feature2`` and ``feature3`` belong to a subset called "A" and ``feature1`` and ``feature1`` and ``feature2`` belong to a subset called "B".

This ``.csv`` file can be provided to ``rsmtool`` using the :ref:`feature_subset_file <feature_subset_file>` field in the configuration file. Then, to select a particular pre-defined subset of features, you simply set the :ref:`feature_subset  <feature_subset>` field in the configuration file to the name of the subset that you wish to use.

Then, in order to use feature subset "A" (``feature2`` and ``feature3``) in an experiment, we need to set the following two fields in our experiment configuration file:

.. code-block:: javascript

    {
        ...
        "feature_subset_file": "subset.csv",
        "feature_subset": "A",
        ...
    }


Transformations
"""""""""""""""
Unlike in :ref:`fine-grained selection <json_column_selection>`, the subset ``.csv`` file does not list any transformations to be applied to the feature columns. However, you can automatically select transformation for each feature *in the selected subset* by applying all possible transforms and identifying the one which gives the highest correlation with the human score. To use this functionality set the :ref:`select_transformations <select_transformations_rsmtool>` field in the configuration file to ``true``.

Signs
"""""
Some guidelines for building scoring models require all coefficients in the model to be positive and all features to have a positive correlation with human score. ``rsmtool`` can automatically flip the sign for any pre-defined feature subset. To use this functionality, the feature subset ``.csv`` file should provide the expected correlation sign between each feature and human score under a column called ``sign_<SUBSET>`` where ``<SUBSET>`` is the name of the feature subset. Then, to tell ``rsmtool`` to flip the the sign for this subset, you need to set the :ref:`sign <sign>` field in the configuration file to ``<SUBSET>``.

To understand this, let's re-examine our earlier example of a subset definition file ``subset.csv``, but with an additional column.

.. code-block:: text

    Feature,A,B,sign_A
    feature1,0,1,+
    feature2,1,1,-
    feature3,1,0,+

Then, in order to use feature subset "A" (``feature2`` and ``feature3``) in an experiment with the sign of ``feature3`` flipped appropriately (multiplied by -1) to ensure positive correlations with score, we need to set the following three fields in our experiment configuration file:

.. code-block:: javascript

    {
        ...
        "feature_subset_file": "subset.csv",
        "feature_subset": "A",
        "sign": "A"
        ...
    }


.. note::

    1. If :ref:`select_transformations <select_transformations_rsmtool>` is set to ``true``, ``rsmtool`` is intelligent enough to take it into account when flipping the signs. For example, if the expected correlation sign for a given feature is negative, ``rsmtool`` will multiply the feature values by ``-1`` if the ``sqrt`` transform has the highest correlation with score. However, if the best transformation turns out to be ``inv`` -- which already changes the polarity of the feature -- no such multiplication will take place.

    2. If you use subset-based feature selection, ``rsmtool`` will create a new ``.json`` file in the ``feature`` folder in the experiment output directory listing all features, signs and transformation as used in the final model. This file will only contain information about the features included in the final model.
