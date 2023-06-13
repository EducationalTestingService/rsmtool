.. _tutorial_rsmexplain:

Tutorial
""""""""

For this tutorial, you first need to :ref:`install RSMTool <install>` and make sure the conda environment in which you installed it is activated.

Workflow
~~~~~~~~

``rsmexplain`` is designed to explain the predictions from a model trained as part of an existing ``rsmtool`` experiment. The steps to do this are as follows:

1. Successfully run an :ref:`rsmtool <usage_rsmtool>` experiment so that the model we would like to explain is trained and available.
2. Create an :ref:`experiment configuration file <config_file_rsmexplain>` describing the explanation experiment you would like to run.
3. Run the created configuration file with :ref:`rsmexplain <usage_rsmexplain>` to generate the explanation HTML report.
4. Examine the HTML report to see the explanations for the ``rsmtool`` model on the selected responses.

ASAP Example
~~~~~~~~~~~~
We are going to use the same example from 2012 Kaggle competition on automated essay scoring that we used for the :ref:`rsmtool tutorial <tutorial>`.

Run ``rsmtool`` experiments with chosen model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``rsmexplain`` requires an existing ``rsmtool`` experiment with a trained model. For this tutorial, we will explain the model trained as part of the :ref:`rsmtool tutorial <tutorial>`.

.. note::

    If you have not already completed the ``rsmtool`` tutorial, please do so now. You may need to complete it again if you deleted the output files.

Create a configuration file
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The next step is to create an :ref:`experiment configuration file <config_file_rsmexplain>` for ``rsmexplain`` in ``.json`` format.

.. _asap_config_rsmexplain:

.. literalinclude:: ../examples/rsmexplain/config_rsmexplain.json
    :language: javascript
    :linenos:

Let's take a look at the options in our configuration file.

- **Line 2**: We give a short description of this experiment. This will be shown in the report.
- **Line 3**: We give the path to the directory containing the output of the original `rsmtool`` experiment. Note that this is the top-level directory that contains the ``output`` directory produced by ``rsmtool``.
- **Line 4**: We provide the ``experiment_id`` of the rsmtool experiment used to train the model. This can usually be read off the ``output/<experiment_id>.model`` file in the rsmtool experiment output directory.
- **Line 5**: We provide the path to the data file that will be used as the background distribution.
- **Line 6**: We provide the path to the data file that will be used to generate the explanations.
- **Line 7**: This field indicates that the unique IDs for the responses in both data files are located in a column named ``ID``.
- **Line 8**: This field indicates that we wish to explain a random sample of 10 responses from the second data file..
- **Line 9**: This field indicates the number of top features that should be displayed in the plots in the ``rsmexplain`` report.
- **Line 10**: This field indicates whether to show the `automatically generated Cohort bar plots <https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/bar.html#Cohort-bar-plot>`_ in the report.

Documentation for all of the available configuration options is available :ref:`here <config_file_rsmexplain>`.

.. note:: You can also use our nifty capability to :ref:`automatically generate <autogen_configuration>` ``rsmexplain`` configuration files rather than creating them manually.

Run explanation experiment
~~~~~~~~~~~~~~~~~~~~~~~~~~
Now that we have the ``rsmtool`` experiment, the data files, and our configuration file, we can use the :ref:`rsmexplain <usage_rsmexplain>` command-line script to run our explanation experiment.

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

1. The various absolute value variants of the SHAP values.
2. Several SHAP plots indicating how different features contribute to the predicted score.
