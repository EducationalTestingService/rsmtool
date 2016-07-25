.. _usage_rsmcompare:

``rsmcompare`` - Compare scoring models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

RSMTool provides the ``rsmcompare`` command-line utility to compare two models and to generate a comparison report. This can be useful in many scenarios, e.g., say the user wants to compare the changes in model performance after adding a new feature into the model. To use ``rsmcompare``, the user must first run two experiments using either :ref:`rsmtool <usage_rsmtool>` or :ref:`rsmeval <usage_rsmeval>`. ``rsmcompare`` can then be used to compare the outputs of these two experiments to each other.

.. note::

    Currently ``rsmcompare`` takes the outputs of the analyses generated during the original experiments and creates comparison tables. These comparison tables were designed with a specific comparison scenario in mind: comparing a baseline model with a model which includes new feature(s). The tool can certianly be used for other comparison scenarios if the researcher feels that the generated comparison output is appropriate.

``rsmcompare`` can be used to compare:

1. Two ``rsmtool`` experiments, or
2. Two ``rsmeval`` experiments, or
3. An ``rsmtool`` experiment with an ``rsmeval`` experiment (in this case, only the evaluation analyses will be compared).


.. note::

    It is strongly recommend that the original experiments as well as the comparison experiment are all done using the same version of RSMTool.

Input
"""""

``rsmcompare`` requires a single argument to run an experiment: the path to a configuration file. It can also take an output directory as an optional second argument. If the latter is not specified, ``rsmcompare`` will use the current directory as the output directory.

.. include:: config_rsmcompare.rst

Output
""""""

``rsmcompare`` produces the comparison report in HTML format as well as in the form of a Jupyter notebook (a ``.ipynb`` file) in the output directory.
