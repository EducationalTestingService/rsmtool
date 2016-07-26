.. _tutorial_rsmcompare:

Tutorial
""""""""

To use the ``rsmcompare`` you need to :ref:`install RSMTool <install>`.

Workflow
~~~~~~~~

``rsmcompare`` is designed to compare two existing ``rsmtool`` experiments. To use ``rsmcompare`` you need:


1. Run two experiments using :ref:`rsmtool <usage_rsmtool>` or :ref:`rsmeval <usage_rsmeval>`. 
2. Create an :ref:`experiment configuration file <config_file_rsmcompare>` describing the comparison experiment you would like to run.
3.  Run that configuration file with :ref:`rsmcompare <usage_rsmcompare>` and generate the comparison experiment HTML report.
4. Examine HTML report to compare the two models.

Note that the above workflow does not use the customization features of ``rsmcompare``, e.g., :ref:`choosing which sections to include in the report <general_sections_rsmcompare>` or :ref:`adding custom analyses sections <custom_notebooks>` etc. However, we will stick with this workflow for our tutorial since it is likely to be the most common use case.

ASAP Example
~~~~~~~~~~~~

We are going to use the same example from 2012 Kaggle competition on automated essay scoring that we used for :ref:`rsmtool tutorial <tutorial>`. 

Run ``rsmtool`` experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~
``rsmcompare`` compares the results of the two existing ``rsmtool`` or ``rsmeval`` experiments. For this tutorial we will compare the outputs generated during the experiment in :ref:`rsmtool tutorial <tutorial>` to itself. If you have not already completed that tutorial first, please do so now. We will be using the output generated during the tutorial, so you will also need to complete the tutorial if you have already deleted the output files. 


Create a configuration file
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The next step is to create an :ref:`rsmcompare experiment configuration file <config_file_rsmcompare>` in ``.json`` format.

.. _asap_config_rsemcompare:

.. literalinclude:: ../examples/rsmcompare/config_rsmcompare.json
    :language: javascript
    :linenos:

Let's take a look at the options in our configuration file.

- **Line 2**: We provide the ``experiment_id`` for the experiment we want to use as a baseline. 
- **Line 3**: We also give the path to the directory containing the output of the original "baseline" experiment. 
- **Line 4**: We give a short description of this experiment. This will be shown in the report. 
- **Line 5**: This field indicates that the original experiment used scaled scores for detailed analyses. 
- **Line 6**: We provide the ``experiment_id`` for the new experiment. In our case we give the same id because we are comparing the experiment to itself. 
- **Line 7**: We also give the path to the directory containing the output of the new experiment. As above, we give the same path because we are comparing the experiment to itself- **Line 8**: We give a short description of this experiment. This will be shown in the report. 
- **Line 9**: This field indicates that the original experiment used scaled scores for detailed analyses. 

Documentation for all of the available configuration options is available :ref:`here <config_file_rsmcompare>`.

Run the experiment
~~~~~~~~~~~~~~~~~~
Now that we have our scores in ``.csv`` format and our configuration file in ``.json`` format, we can use the :ref:`rsmcompare <usage_rsmcompare>` command-line script to run our modeling experiment.

.. code-block:: bash

    $ cd examples/rsmcompare
    $ rsmtool config_rsmcompare.json

This should produce output like::

    Output directory: /home/aloukina/proj/rsmtool/rsmtool-github/examples/rsmcompare
    Starting report generation
    Merging sections
    Exporting HTML
    Executing notebook with kernel: python3


Once the run finishes, you will see the a report named ``ASAP2_vs_ASAP2.report.html``. This is the final ``rsmcompare`` comparison report.

Examine the report
~~~~~~~~~~~~~~~~~~
Our experiment report contains all the information we would need to evaluate the system scores against the human scores. It includes:

1. Comparison of feature distributions including differences between the two models. 
2. Comparison of model coefficients including differences between the two models.
3. Comparison of model performance including differences between the two models.

.. note::
    Since we are comparing the experiment to itself, there is no change between the models and therefore the values are always 0.



