Tutorial
========

First you'll want to :ref:`install RSMTool <install>`.

Workflow
--------

.. important::

    Although this tutorial provides feature values for the purposes of illustration, RSMTool is *not* designed to extract features; it is designed for researchers who use their own NLP/Speech processing pipeline to extract features for their data.

Once you have the features extracted your data, using RSMTool is fairly straightforward:

1. Make sure your data is split into a training set and a held-out evaluation set. You will need two ``.csv`` files containing the feature values: one for each set.
2. Create an :ref:`experiment configuration file <config_file_rsmtool>` describing the modeling experiment you would like to run.
3.  Run that configuration file with :ref:`rsmtool <usage_rsmtool>` and generate the experiment HTML report as well as the :ref:`intermediate CSV files <intermediate_files_rsmtool>`.
4. Examine HTML report to check various aspects of model performance.

Note that the above workflow does not use the customization features of RSMTool, e.g., :ref:`choosing which sections to include in the report <general_sections_rsmtool>` or :ref:`adding custom analyses sections <custom_notebooks>` etc. However, we will stick with this workflow for our tutorial since it is likely to be the most common use case.

ASAP Example
------------

Let's see how we can apply this basic RSMTool workflow to a simple example based on a 2012 Kaggle competition on automated essay scoring called the `Automated Student Assessment Prize (ASAP) contest <https://www.kaggle.com/c/asap-aes>`_. As part of that contest, responses to 8 different essay questions written by students in grades 6-10 were provided. The responses were scored by humans and given a holistic score indicating the English proficiency of the student. The goal of the contest was to build an automated scoring model with the highest accuracy on held-out evaluation data.

For our tutorial, we will use one of the questions from this data to illustrate how to use RSMTool to train a model and evaluate its performance.

Extract features
^^^^^^^^^^^^^^^^
Kaggle provides the actual text of the responses along with the human scores in a ``.tsv`` file. The first step, as with any automated scoring approaches, is to extract features from the written (or spoken) responses that might be useful in predicting the score we are interested in. For example, some features that might be useful for predicting the English proficiency of a response are:

- Does the response contain any grammatical errors and if so, how many and of what kind since some grammatical errors are more serious than others.
- Is the response is well organized, e.g., whether there is a clear thesis statement and a conclusion, etc.
- Is the response coherent, i.e., do the ideas expressed in the various paragraphs well connected?
- Does the response contain any spelling errors?

For more examples of what features might look like, we refer the reader to this paper by Attali & Burstein [#]_. For our ASAP2 data, we extracted the following four features:

- ``GRAMMAR``: indicates how grammatically fluent the responses is.
- ``MECHANICS``: indicates how free of mechanical errors (e.g., spelling errors) the response is.
- ``DISCOURSE``: indicates whether the discourse transitions in the response make sense.
- ``ORGANIZATION``: indicates whether the response is organized well.

.. note::

     These features were extracted using proprietary NLP software. To reiterate, RSMTool does *not* include any NLP/speech components for feature extraction.


Create a configuration file
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The next step is to create an :ref:`RSMTool experiment configuration file <config_file_rsmtool>` in ``.json`` format.

.. _asap_config:

.. literalinclude:: ../examples/rsmtool/config_rsmtool.json
    :language: javascript
    :linenos:

Let's take a look at the options in our configuration file.

- **Line 2**: We define an experiment ID
- **Lines 3-4**: We list the paths to our training and evaluation ``.csv`` files with the feature values.
- **Line 5**: We choose to use a linear regression model to combine those features into a score.
- **Line 6**: We also provide a description which will be included in the experiment report.
- **Lines 7-8**: These two fields indicate that the human scores in the two ``.csv`` files are located in columns named ``score``.
- **Lines 9**: Next, we indicate that we would like to use the :ref:`scaled scores <score_postprocessing>` for our evaluation analyses.
- **Lines 10-11**: These fields indicate that the lowest score on the scoring scale is a 1 and the highest score is a 6. This information is usually part of the rubric used by human graders.
- **Line 12**: This field indicates that the unique IDs for the responses in the two ``.csv`` files are located in a column named ``ID``.
- **Line 13**: This field indicates that scores from a second set of human graders are also available (useful for comparing the agreement between human-machine scores to the agreement between two sets of humans) and are located in the ``score2`` column in the test set ``.csv`` file.
- **Line 14**: This field indicates that response lengths are also available for the training data in a column named ``LENGTH``.

Documentation for all of the available configuration options is :ref:`available <config_file_rsmtool>`.

.. note::

    For this example, we are using *all* of the non-metadata columns in the training and evaluation ``.csv`` files as features in the model. However, it is also possible to :ref:`choose specific columns <column_selection_rsmtool>` to be used for training the model.

Run the experiment
^^^^^^^^^^^^^^^^^^
Now that we have our features in ``.csv`` format and our configuration file in ``.json`` format, we can use the :ref:`rsmtool <usage_rsmtool>` command-line script to run our modeling experiment.

.. code-block:: bash

    $ cd examples/rsmtool
    $ rsmtool config_rsmtool.json

This should produce output like::

    Output directory: /Users/nmadnani/work/rsmtool/examples/rsmtool
    Loading experiment data
    Reading configuration file: /Users/nmadnani/work/rsmtool/examples/rsmtool/config_rsmtool.json
    Reading training data: /Users/nmadnani/work/rsmtool/examples/rsmtool/train.csv
    Reading evaluation data: /Users/nmadnani/work/rsmtool/examples/rsmtool/test.csv
    Pre-processing training and test set features
    Saving training and test set data to disk
    Training LinearRegression model
    The exact solution is  x = 0
    Running analyses on training set
    Generating training and test set predictions
    Processing test set predictions
    Saving training and test set predictions to disk
    Scaling the coefficients and saving them to disk
    Running analyses on test set predictions
    Starting report generation
    Merging sections
    Exporting HTML
    Executing notebook with kernel: python3




.. rubric:: References

.. [#] Attali, Y., & Burstein, J. (2006). Automated essay scoring with e-rater® V.2. Journal of Technology, Learning, and Assessment, 4(3). https://ejournals.bc.edu/ojs/index.php/jtla/article/download/1650/1492
