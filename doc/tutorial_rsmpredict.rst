.. _tutorial_rsmpredict

Tutorial
""""""""

To use the ``rsmpredict`` you need to :ref:`install RSMTool <install>`.

Workflow
~~~~~~~~

.. important::

    Although this tutorial provides feature values for the purpose of illustration, ``rsmpredict`` does *not* include any functionality for feature extraction; the tool is :ref:`designed for researchers <who_rsmtool>` who use their own NLP/Speech processing pipeline to extract features for their data.


``rsmpredict`` allows you to generate the scores for new data using an existing model trained by RSMTool. Therefore before attempting this tutorial you need to train an RSMTool model by completing :ref:`rsmtool tutorial <tutorial>`. You will also need to process the new data to extract the same features as the ones used in the model. 

Once you have the features for the new data and the RSMTool model, using ``rsmpredict`` is fairly straightforward:  

1. Create a ``.csv`` file containing the features for the new data. 
2. Create an :ref:`experiment configuration file <config_file_rsmpredict>` describing the  experiment you would like to run.
3.  Run that configuration file with :ref:`rsmpredict <usage_rsmpredict>` to generate the scores.

ASAP Example
~~~~~~~~~~~~

We are going to use the same example from 2012 Kaggle competition on automated essay scoring that we used for :ref:`rsmtool tutorial <tutorial>`. If you have not already completed that tutorial first, please do so now. We will be using the output generated during the tutorial, so you will also need to complete the tutorial if you have already deleted the output files.

We are now going to use this model to generate scores for new data.   

Extract features
^^^^^^^^^^^^^^^^
We will first need to generate features for the new set of responses for which we need to generate system scores.  For this experiment we will simply re-use the test set from our RSMTool tutorial. 

.. note::

    The features used with ``rsmpredict`` should be generated using the same NLP/Speech processing pipeline as used to generate features for the original model building experiment. 

Create a configuration file
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The next step is to create an :ref:`rsmpredict experiment configuration file <config_file_rsmpredict>` in ``.json`` format.

.. _asap_config_rsempredict:

.. literalinclude:: ../examples/rsmpredict/config_rsmpredict.json
    :language: javascript
    :linenos:

Let's take a look at the options in our configuration file.

- **Line 2**: We provide the ``experiment_id`` for the experiment used to create the model files. 
- **Line 3**: We give the path to the directory containing the output of the original experiment. 
- **Line 4**: We list the path to the ``.csv`` with features for the new data.
- **Line 5**: This field indicates that the unique IDs for the responses in the two ``.csv`` files are located in a column named ``ID``.
- **Lines 6-7**: These fields indicates that there are two sets of human scores in our ``.csv`` file located in the columns named ``score`` and ``score2``. The values from these columns will be added to the prediction file. 

Documentation for all of the available configuration options is available :ref:`here <config_file_rsmpredict>`.


.. note::

    You do not need to have human scores to run ``rsmpredict`` since it does not produce any evaluations. If you have human scores and would like to conduct the evaluations of the new scores, you can follow these steps:
         1. Add the score column to the configuration file as shown above. 
         2. Run :ref:`rsmeval <usage_rsmeval>` to produce evaluation report for these scores.  

Run the experiment
~~~~~~~~~~~~~~~~~~
Now that we have the model, the features in ``.csv`` format and our configuration file in ``.json`` format, we can use the :ref:`rsmpredict <usage_rsmpredict>` command-line script to generate the predictions and to save them in ``predictions.csv``.

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

You will now see a file named ``predictions.csv`` which contains the predicted scores for the new data. 

Prediction file columns
~~~~~~~~~~~~~~~~~~~~~~~
The predictions file contains the following columns:

    - ``spkitemid`` - the unique resonse IDs copied from the ``ID`` column in the original feature file
    - ``sc1``, ``sc2`` - the human score columns copied from the original feature file.The  ``human_score_column`` (``score``) was renamed to ``sc1`` and the ``second_human_score_column`` (``score2``) was renamed to ``sc2``. See the :ref:`documentation <config_file_rsmpredict>` for the naming conventions. 
    - ``raw`` - raw scores generated by applying the model to the new features
    - ``raw_trim``, ``raw_trim_round``, ``scale``, ``scale_trim``, ``scale_trim_round`` - raw scores postprocessed in different ways. See :ref:`documentation <score_postprocessing>` for further detail about different post-processing methods. 


