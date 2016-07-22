.. _usage_rsmcompare:

``rsmcompare`` - Compare scoring models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RSMTool provides the ``rsmcompare`` command-line utility to compare two models and to generate a report with a comparison. This can be useful in scenarios where the user wants to compare the changes in model performance after adding a new feature. To use ``rsmcompare`` the user must first run two experiments using ``rsmtool`` or ``rsmeval``. ``rsmcompare`` can then be used to compare the outputs of these two experiments.  

.. note::
    Currently ``rsmcompare`` takes the outputs of the analyses generated during the original experiments and creates comparison tables. These comparison tables were designed to compare a baseline model with the model which includes new feature. While the tool can be used to evaluate other changes, the researcher needs to make a judgement whether a given comparison is meanigful for their experiments. 

The ``rsmcompare`` can be used to compare two ``rsmtool`` experiments as well as two ``rsmeval`` evaluations or an ``rsmtool`` evaluation with an ``rsmeval`` evaluation. 

.. note::
    We strongly recommend that the original experiments and the comparison are done using the same version of the tool. 

Input
"""""

``rsmcompare`` requires a single argument to run an experiment: the path to a configuration file. It can also take an output directory as an optional second argument. If the latter is not specified, ``rsmcompare`` will use the current directory as the output directory.

.. include:: config_rsmcompare.rst

Output
""""""

``rsmcompare`` produces the comparison report in HTML format as well as in the form of a Jupyter notebook (a ``.ipynb`` file).
