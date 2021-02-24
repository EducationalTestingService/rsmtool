.. _tutorial_rsmcompare:

Tutorial
""""""""

For this tutorial, you first need to :ref:`install RSMTool <install>` and make sure the conda environment in which you installed it is activated.

Workflow
~~~~~~~~

``rsmcompare`` is designed to compare two existing ``rsmtool`` or ``rsmeval`` experiments. To use ``rsmcompare`` you need:

1. Two experiments that were run using :ref:`rsmtool <usage_rsmtool>` or :ref:`rsmeval <usage_rsmeval>`.
2. Create an :ref:`experiment configuration file <config_file_rsmcompare>` describing the comparison experiment you would like to run.
3.  Run that configuration file with :ref:`rsmcompare <usage_rsmcompare>` and generate the comparison experiment HTML report.
4. Examine HTML report to compare the two models.

Note that the above workflow does not use the customization features of ``rsmcompare``, e.g., :ref:`choosing which sections to include in the report <general_sections_rsmcompare>` or :ref:`adding custom analyses sections <custom_notebooks>` etc. However, we will stick with this workflow for our tutorial since it is likely to be the most common use case.

ASAP Example
~~~~~~~~~~~~
We are going to use the same example from 2012 Kaggle competition on automated essay scoring that we used for the :ref:`rsmtool tutorial <tutorial>`.

Run ``rsmtool`` (or ``rsmeval``) experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``rsmcompare`` compares the results of the two existing ``rsmtool`` (or ``rsmeval``) experiments. For this tutorial, we will compare model trained in the :ref:`rsmtool tutorial <tutorial>` to itself.

.. note::

    If you have not already completed that tutorial, please do so now. You may need to complete it again if you deleted the output files.

Create a configuration file
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The next step is to create an :ref:`experiment configuration file <config_file_rsmcompare>` in ``.json`` format.

.. _asap_config_rsmcompare:

.. literalinclude:: ../examples/rsmcompare/config_rsmcompare.json
    :language: javascript
    :linenos:

Let's take a look at the options in our configuration file.

- **Line 2**: We provide an ID for the comparison experiment.
- **Line 3**: We provide the ``experiment_id`` for the experiment we want to use as a baseline.
- **Line 4**: We also give the path to the directory containing the output of the original baseline experiment.
- **Line 5**: We give a short description of this baseline experiment. This will be shown in the report.
- **Line 6**: This field indicates that the baseline experiment used scaled scores for some evaluation analyses.
- **Line 7**: We provide the ``experiment_id`` for the new experiment. We use the same experiment ID for both experiments since we are comparing the experiment to itself.
- **Line 8**: We also give the path to the directory containing the output of the new experiment. As above, we use the same path because we are comparing the experiment to itself.
- **Line 9**: We give a short description of the new experiment. This will also be shown in the report.
- **Line 10**: This field indicates that the new experiment also used scaled scores for some evaluation analyses.

Documentation for all of the available configuration options is available :ref:`here <config_file_rsmcompare>`.

.. note:: You can also use our nifty capability to :ref:`automatically generate <autogen_configuration>` ``rsmcompare`` configuration files rather than creating them manually.

Run the experiment
~~~~~~~~~~~~~~~~~~
Now that we have the two experiments we want to compare and our configuration file in ``.json`` format, we can use the :ref:`rsmcompare <usage_rsmcompare>` command-line script to run our comparison experiment.

.. code-block:: bash

    $ cd examples/rsmcompare
    $ rsmcompare config_rsmcompare.json

This should produce output like::

    Output directory: /Users/nmadnani/work/rsmtool/examples/rsmcompare
    Starting report generation
    Merging sections
    Exporting HTML
    Executing notebook with kernel: python3

Once the run finishes, you will see an HTML file named ``ASAP2_vs_ASAP2_report.html``. This is the final ``rsmcompare`` comparison report.

Examine the report
~~~~~~~~~~~~~~~~~~
Our experiment report contains all the information we would need to compare the new model to the baseline model. It includes:

1. Comparison of feature distributions between the two experiments.
2. Comparison of model coefficients between the two experiments.
3. Comparison of model performance between the two experiments.

.. note::

    Since we are comparing the experiment to itself, the comparison is not very interesting, e.g., the differences between various values will always be 0.
