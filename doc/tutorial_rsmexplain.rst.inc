.. _tutorial_rsmexplain:

Tutorial
""""""""

For this tutorial, you first need to :ref:`install RSMTool <install>` and make sure the conda environment in which you installed it is activated.

Workflow
~~~~~~~~

``rsmexplain`` is designed to explain an existing ``rsmtool`` experiment. To use ``rsmexplain`` you need:

1. One experiment that was run using :ref:`rsmtool <usage_rsmtool>`.
2. Create an :ref:`experiment configuration file <config_file_rsmexplain>` describing the explanation experiment you would like to run.
3. Run that configuration file with :ref:`rsmexplain <usage_rsmexplain>` and generate the explanation experiment HTML report.
4. Examine HTML report to explain the non-linear model.

ASAP Example
~~~~~~~~~~~~
We are going to use the same example from 2012 Kaggle competition on automated essay scoring that we used for the :ref:`rsmtool tutorial <tutorial>`.

Run ``rsmtool`` experiments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``rsmexplain`` explains the results of the existing ``rsmtool`` experiment.  For this tutorial, we will explain model trained in the :ref:`rsmtool tutorial <tutorial>` to itself.

.. note::

    If you have not already completed that tutorial, please do so now. You may need to complete it again if you deleted the output files.

Create a configuration file
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The next step is to create an :ref:`experiment configuration file <config_file_rsmexplain>` in ``.json`` format.

.. _asap_config_rsmexplain:

.. literalinclude:: ../examples/rsmexplain/config_rsmexplain.json
    :language: javascript
    :linenos:

Let's take a look at the options in our configuration file.

- **Line 2**: We give a short description of this experiment. This will be shown in the report.
- **Line 3**: We give the path to the directory containing the output of the original experiment.
- **Line 4**: We provide the experiment_id of the rsmtool experiment used to train the model. This can usually be read off the output/<experiment_id>.model file in the rsmtool experiment output directory.
- **Line 5**: We list the path to the data file for use as the background distribution.
- **Line 6**: We list the path to the data file to explain.
- **Line 7**: This field indicates that the unique IDs for the responses in the ``.csv`` file are located in a column named ``ID``.
- **Line 8**: This field indicates taht size of random sample to be explained.
- **Line 9**: This field indicates number of top features should be displayed in rsmexplain plots.
- **Line 10**: This feild indicates whether to show auto cohorts.

Documentation for all of the available configuration options is available :ref:`here <config_file_rsmexplain>`.

.. note:: You can also use our nifty capability to :ref:`automatically generate <autogen_configuration>` ``rsmexplain`` configuration files rather than creating them manually.

Run the experiment
~~~~~~~~~~~~~~~~~~
Now that we have the model, the features in the right format, and our configuration file in ``.json`` format, we can use the :ref:`rsmexplain <usage_rsmexplain>` command-line script to run our explanation experiment.

.. code-block:: bash

    $ cd examples/rsmexplain
    $ rsmexplain config_rsmexplain.json

This should produce output like::

    Output directory: /Users/nmadnani/work/rsmtool/examples/rsmexplain
    Generating shap explanations for 355 examples from explain_features.csv
    Merging sections
    Exporting HTML
    Success

Once the run finishes, you will see the ``output``, ``figure``, and ``report`` sub-directories in the current directory. Each of these directories contain :ref:`useful information <output_dirs_rsmexplain>` but we are specifically interested in the ``report/ASAP2_explain_report.html`` file, which is the final evaluation report.

Examine the report
~~~~~~~~~~~~~~~~~~
Our experiment report contains all the information we would need to explain the trained model. It includes:

1. The various absolute value variants of the SHAP vlaues.
2. Several SHAP plots indicating how different features contribute to the final target.
