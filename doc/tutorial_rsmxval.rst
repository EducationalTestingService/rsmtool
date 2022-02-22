.. _tutorial_rsmxval:

Tutorial
""""""""

For this tutorial, you first need to :ref:`install RSMTool <install>` and make sure the conda environment in which you installed it is activated.

Workflow
~~~~~~~~

``rsmxval`` is designed to run cross-validation experiments using a single file containing human scores and features. Just like ``rsmtool``, ``rsmxval`` does not provide any functionality for feature extraction and assumes that users will extract features on their own. The workflow steps are as follows:

1. Create a data file in one of the :ref:`supported formats <input_file_format>` containing the extracted features for each response in the data along with human score(s) assigned to it.
2. Create an :ref:`experiment configuration file <config_file_rsmxval>` describing the cross-validation experiment you would like to run.
3. Run that configuration file with :ref:`rsmxval <usage_rsmxval>` and generate its :ref:`outputs <output_dirs_rsmxval>`.
4. Examine the various HTML reports to check various aspects of model performance.

Note that unlike ``rsmtool`` and ``rsmeval``, ``rsmxval`` currently does not support customization of the HTML reports generated in each step. This functionality may be added in future versions.

ASAP Example
~~~~~~~~~~~~
We are going to use the same example from 2012 Kaggle competition on automated essay scoring that we used for the :ref:`rsmtool tutorial <tutorial>`.

Extract features
~~~~~~~~~~~~~~~~
We are using the same features for this data as described in the :ref:`rsmtool tutorial <tutorial>`.

Create a configuration file
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The next step is to create an :ref:`experiment configuration file <config_file_rsmxval>` in ``.json`` format.

.. _asap_config_rsmxval:

.. literalinclude:: ../examples/rsmxval/config_rsmxval.json
    :language: javascript
    :linenos:

Let's take a look at the options in our configuration file.

- **Line 2**: We define an experiment ID used to identify the files produced as part of this experiment.
- **Line 3**: We also provide a description which will be included in the various reports. 
- **Line 4**: We list the path to our training file with the feature values and human scores.  For this tutorial, we used ``.csv`` format, but several other :ref:`input file formats <input_file_format>` are also supported.
- **Line 5**: This field indicates the number of cross-validation folds we want to use. If this field is not specified, ``rsmxval`` uses 5-fold cross-validation by default.
- **Line 6**: This field indicates that the human (reference) scores in our ``.csv`` file are located in a column named ``score``.
- **Line 7**: This field indicates that the unique IDs for the responses in the ``.csv`` file are located in a column named ``ID``.
- **Line 8**: We choose to use a linear regression model to combine the feature values into a score.
- **Lines 9-10**: These fields indicate that the lowest score on the scoring scale is a 1 and the highest score is a 6. This information is usually part of the rubric used by human graders.
- **Line 11**: This field indicates that scores from a second set of human graders are also available (useful for comparing the agreement between human-machine scores to the agreement between two sets of humans) and are located in the ``score2`` column in the training ``.csv`` file.
- **Line 12**: Next, we indicate that we would like to use the :ref:`scaled scores <score_postprocessing>` for all our evaluation analyses at each step.

Documentation for all of the available configuration options is available :ref:`here <config_file_rsmxval>`.

.. note:: You can also use our nifty capability to :ref:`automatically generate <autogen_configuration>` ``rsmxval`` configuration files rather than creating them manually.


Run the experiment
~~~~~~~~~~~~~~~~~~
Now that we have our input file and our configuration file, we can use the :ref:`rsmxval <usage_rsmxval>` command-line script to run our evaluation experiment.

.. code-block:: bash

    $ cd examples/rsmxval
    $ rsmxval config_rsmxval.json output

This should produce output like::

    Output directory: output
    Saving configuration file.
    Generating 3 folds after shuffling
    Running RSMTool on each fold in parallel
    Progress: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:08<00:00,  2.76s/it]
    Creating fold summary
    Evaluating combined fold predictions
    Training model on full data


Once the run finishes, you will see an ``output`` sub-directory in the current directory. Under this directory you will see multiple other sub-directories, each corresponding to one of the cross-validation steps and described :ref:`here <output_dirs_rsmeval>`. 

Examine the reports
~~~~~~~~~~~~~~~~~~~
The cross-validation experiment produces multiple HTML reports - an ``rsmtool`` report for each of the 3 folds (``output/folds/{01,02,03}/report/ASAP2_xval_fold{01,02,03}.html``), the evaluation report for the cross-validated predictions (``output/evaluation/report/ASAP2_xval_evaluation_report.html``), a report summarizing the salient characteristics of the 3 folds (``output/fold-summary/report/ASAP2_xval_fold_summary_report.html``), a report showing the feature and model descriptives (``output/final-model/report/ASAP2_xval_model_report.html``). Examining these reports will provide a relatively complete picture of how well the predictive performance of the scoring model will generalize to unseen data.

