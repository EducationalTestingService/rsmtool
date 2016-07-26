.. _tutorial_rsmpredict:

Tutorial
""""""""

For this tutorial, you first need to :ref:`install RSMTool <install>` and make sure you have activated the ``rsmtool`` conda environment before you start the tutorial.

Workflow
~~~~~~~~

.. important::

    Although this tutorial provides feature values for the purpose of illustration, ``rsmpredict`` does *not* include any functionality for feature extraction; the tool is :ref:`designed for researchers <who_rsmtool>` who use their own NLP/Speech processing pipeline to extract features for their data.

``rsmpredict`` allows you to generate the scores for new data using an existing model trained using RSMTool. Therefore, before starting this tutorial, you first need to complete :ref:`rsmtool tutorial <tutorial>` which will produce a train RSMTool model. You will also need to process the new data to extract the same features as the ones used in the model.

Once you have the features for the new data and the RSMTool model, using ``rsmpredict`` is fairly straightforward:

1. Create a ``.csv`` file containing the features for the new data.
2. Create an :ref:`experiment configuration file <config_file_rsmpredict>` describing the  experiment you would like to run.
3.  Run that configuration file with :ref:`rsmpredict <usage_rsmpredict>` to generate the predicted scores.

    .. note::

        You do not *need* human scores to run ``rsmpredict`` since it does not produce any evaluation analyses. If you do have human scores for the new data and you would like to evaluate the system on this new data, you can first run ``rsmpredict`` to generate the predictions and then run ``rsmeval`` on the output of ``rsmpredict`` to generate an evaluation report.

ASAP Example
~~~~~~~~~~~~
We are going to use the same example from 2012 Kaggle competition on automated essay scoring that we used for the :ref:`rsmtool tutorial <tutorial>`. Specifically, We are going to use the linear regression model we trained in that tutorial to generate scores for new data.

.. note::

    If you have not already completed that tutorial, please do so now. You may need to complete it again if you deleted the output files.

Extract features
~~~~~~~~~~~~~~~~
We will first need to generate features for the new set of responses for which we want to predict scores.  For this experiment, we will simply re-use the test set from the ``rsmtool`` tutorial.

.. note::

    The features used with ``rsmpredict`` should be generated using the *same* NLP/Speech processing pipeline that generated the features used in the ``rsmtool`` modeling experiment.

Create a configuration file
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The next step is to create an :ref:`rsmpredict experiment configuration file <config_file_rsmpredict>` in ``.json`` format.

.. _asap_config_rsempredict:

.. literalinclude:: ../examples/rsmpredict/config_rsmpredict.json
    :language: javascript
    :linenos:

Let's take a look at the options in our configuration file.

- **Line 2**: We give the path to the directory containing the output of the ``rsmtool`` experiment.
- **Line 3**: We provide the ``experiment_id`` of the ``rsmtool`` experiment used to train the model. This can usually be read off the ``output/<experiment_id>.model`` file in the ``rsmtool`` experiment output directory.
- **Line 4**: We list the path to the ``.csv`` file with the feature values for the new data.
- **Line 5**: This field indicates that the unique IDs for the responses in the ``.csv`` file are located in a column named ``ID``.
- **Lines 6-7**: These fields indicates that there are two sets of human scores in our ``.csv`` file located in the columns named ``score`` and ``score2``. The values from these columns will be added to the output file containing the predictions which can be useful if we want to evaluate the predictions using ``rsmeval``.

Documentation for all of the available configuration options is available :ref:`here <config_file_rsmpredict>`.

Run the experiment
~~~~~~~~~~~~~~~~~~
Now that we have the model, the features in ``.csv`` format, and our configuration file in ``.json`` format, we can use the :ref:`rsmpredict <usage_rsmpredict>` command-line script to generate the predictions and to save them in ``predictions.csv``.

.. code-block:: bash

    $ cd examples/rsmpredict
    $ rsmpredict config_rsmpredict.json predictions.csv

This should produce output like::

    WARNING: The following extraenous features will be ignored: {'spkitemid', 'sc1', 'sc2', 'LENGTH'}
    Pre-processing input features
    Generating predictions
    Rescaling predictions
    Trimming and rounding predictions
    Saving predictions to /home/aloukina/proj/rsmtool/rsmtool-github/examples/rsmpredict/predictions.csv

You should now see a file named ``predictions.csv`` in the current directory which contains the predicted scores for the new data in the ``predictions`` column.




