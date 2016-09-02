.. _tutorial_rsmsummarize:

Tutorial
""""""""

For this tutorial, you first need to :ref:`install RSMTool <install>` and make sure the conda environment in which you installed it is activated.

Workflow
~~~~~~~~

``rsmsummarize`` is designed to compare sevearl existing ``rsmtool`` or ``rsmeval`` experiments. To use ``rsmsummarize`` you need:

1. Two or more experiments that were run using :ref:`rsmtool <usage_rsmtool>` or :ref:`rsmeval <usage_rsmeval>`.
2. Create an :ref:`experiment configuration file <config_file_rsmsummarize>` describing the comparison experiment you would like to run.
3. Run that configuration file with :ref:`rsmsummarize <usage_rsmsummarize>` and generate the comparison experiment HTML report.
4. Examine HTML report to compare the models.

Note that the above workflow does not use the customization features of ``rsmsummarize``, e.g., :ref:`choosing which sections to include in the report <general_sections_rsmsummarize>` or :ref:`adding custom analyses sections <custom_notebooks>` etc. However, we will stick with this workflow for our tutorial since it is likely to be the most common use case.

ASAP Example
~~~~~~~~~~~~
We are going to use the same example from 2012 Kaggle competition on automated essay scoring that we used for the :ref:`rsmtool tutorial <tutorial>`.

Run ``rsmtool`` and ``rsmeval`` experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``rsmsummarize`` compares the results of the two or more existing ``rsmtool`` (or ``rsmeval``) experiments. For this tutorial, we will compare model trained in the :ref:`rsmtool tutorial <tutorial>` to the evaluations we obtained in the  :ref:`rsmeval tutorial <tutorial_rsmeval>`.

.. note::

    If you have not already completed these tutorials, please do so now. You may need to complete them again if you deleted the output files.

Create a configuration file
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The next step is to create an :ref:`experiment configuration file <config_file_rsmsummarize>` in ``.json`` format.

.. _asap_config_rsmsummarize:

.. literalinclude:: ../examples/rsmsummarize/config_rsmsummarize.json
    :language: javascript
    :linenos:

Let's take a look at the options in our configuration file.

- **Line 2**: We provide the ``summary_id`` for the comparison
- **Line 3**: We give a short description of this comparison experiment. This will be shown in the report.
- **Line 4**: We also give the list of paths to the directories containing the outputs of the experiments we want to compare.

Documentation for all of the available configuration options is available :ref:`here <config_file_rsmsummarize>`.

Run the experiment
~~~~~~~~~~~~~~~~~~
Now that we have the list of the experiments we want to compare and our configuration file in ``.json`` format, we can use the :ref:`rsmsummarize <usage_rsmsummarize>` command-line script to run our comparison experiment.

.. code-block:: bash

    $ cd examples/rsmsummarize
    $ rsmsummarize config_rsmsummarize.json

This should produce output like::

    Output directory: /Users/nmadnani/work/rsmtool/examples/rsmsummarize
    Starting report generation
    Merging sections
    Exporting HTML
    Executing notebook with kernel: python3

Once the run finishes, you will see an HTML file named ``model_comparison_report.html``. This is the final ``rsmsummarize`` summary report.

Examine the report
~~~~~~~~~~~~~~~~~~
Our experiment report contains the overview of main aspects of model performance. It includes:

1. Brief description of all experiments.
2. Information about model parameters and model fit for all ``rsmtool`` experiments.
3. Model performance for all experiments. 


.. note::

    Some of the information such as model fit and model parameters are only available for ``rsmtool`` experiments. 


