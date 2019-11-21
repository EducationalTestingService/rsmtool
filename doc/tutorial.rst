.. _tutorial:

Tutorial
========

First you'll want to :ref:`install RSMTool <install>` and make sure you the conda environment in which you installed it is activated.

Workflow
--------

.. important::

    Although this tutorial provides feature values for the purpose of illustration, RSMTool does *not* include any functionality for feature extraction; the tool is :ref:`designed for researchers <who_rsmtool>` who use their own NLP/Speech processing pipeline to extract features for their data.

Once you have the features extracted from your data, using RSMTool is fairly straightforward:

1. Make sure your data is split into a training set and a held-out evaluation set. You will need two files containing the feature values: one for each set. The files should be in one of the :ref:`supported formats <input_file_format>`
2. Create an :ref:`experiment configuration file <config_file_rsmtool>` describing the modeling experiment you would like to run.
3.  Run that configuration file with :ref:`rsmtool <usage_rsmtool>` and generate the experiment HTML report as well as the :ref:`intermediate CSV files <intermediate_files_rsmtool>`.
4. Examine the HTML report to check various aspects of model performance.

Note that the above workflow does not use the customization features of RSMTool, e.g., :ref:`choosing which sections to include in the report <general_sections_rsmtool>` or :ref:`adding custom analyses sections <custom_notebooks>` etc. However, we will stick with this workflow for our tutorial since it is likely to be the most common use case.

ASAP Example
------------

Let's see how we can apply this basic RSMTool workflow to a simple example based on a 2012 Kaggle competition on automated essay scoring called the `Automated Student Assessment Prize (ASAP) contest <https://www.kaggle.com/c/asap-aes>`_. As part of that contest, responses to 8 different essay questions written by students in grades 6-10 were provided. The responses were scored by humans and given a holistic score indicating the English proficiency of the student. The goal of the contest was to build an automated scoring model with the highest accuracy on held-out evaluation data.

For our tutorial, we will use one of the questions from this data to illustrate how to use RSMTool to train a scoring model and evaluate its performance. All of the data we refer to below can be found in the ``examples/rsmtool`` folder in the `github repository <https://github.com/EducationalTestingService/rsmtool/tree/master/examples/rsmtool>`_.

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
- **Lines 3-4**: We list the paths to our training and evaluation files with the feature values.  For this tutorial we used ``.csv`` format, but RSMTool also supports several other :ref:`input file formats <input_file_format>`.
- **Line 5**: We choose to use a linear regression model to combine those features into a score.
- **Line 6**: We also provide a description which will be included in the experiment report.
- **Lines 7-8**: These two fields indicate that the human scores in the two ``.csv`` files are located in columns named ``score``.
- **Lines 9**: Next, we indicate that we would like to use the :ref:`scaled scores <score_postprocessing>` for our evaluation analyses.
- **Lines 10-11**: These fields indicate that the lowest score on the scoring scale is a 1 and the highest score is a 6. This information is usually part of the rubric used by human graders.
- **Line 12**: This field indicates that the unique IDs for the responses in the two ``.csv`` files are located in columns named ``ID``.
- **Line 13**: This field indicates that scores from a second set of human graders are also available (useful for comparing the agreement between human-machine scores to the agreement between two sets of humans) and are located in the ``score2`` column in the test set ``.csv`` file.
- **Line 14**: This field indicates that response lengths are also available for the training data in a column named ``LENGTH``.

Documentation for all of the available configuration options is available :ref:`here <config_file_rsmtool>`.

.. note::

    For this example, we are using *all* of the non-metadata columns in the training and evaluation ``.csv`` files as features in the model. However, it is also possible to :ref:`choose specific columns <column_selection_rsmtool>` to be used for training the model.

Run the experiment
^^^^^^^^^^^^^^^^^^
Now that we have our features in right format and our configuration file in ``.json`` format, we can use the :ref:`rsmtool <usage_rsmtool>` command-line script to run our modeling experiment.

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

Once the run finishes, you will see the ``output``, ``figure``, ``report``, and ``feature`` sub-directories in the current directory. Each of these directories contain :ref:`useful information <output_dirs_rsmtool>` but we are specifically interested in the ``report/ASAP2_report.html`` file, which is the final RSMTool experiment report.

Examine the report
^^^^^^^^^^^^^^^^^^
Our experiment report contains all the information we would need to determine how well our linear regression model with four features is doing. It includes:

1. Descriptive feature statistics.
2. Inter-feature correlations.
3. Model fit and diagnostics.
4. Several different metrics indicating how well the machine's scores agree with the humans'.

... and :ref:`much more <general_sections_rsmtool>`.

If we are not satisfied with the performance of the model, we can then take specific action, e.g., either add new features or perhaps try a more sophisticated model. And then run ``rsmtool`` again and examine the report. Building a scoring model is generally an iterative process.

What's next?
------------
Next, you should read the detailed documentation on :ref:`rsmtool <usage_rsmtool>` since the tutorial only covered a very basic example. If you are interested in RSMTool but your use case is more nuanced, we have probably got you :ref:`covered <advanced_usage>`.

.. rubric:: References

.. [#] Attali, Y., & Burstein, J. (2006). Automated essay scoring with e-rater® V.2. Journal of Technology, Learning, and Assessment, 4(3). https://ejournals.bc.edu/index.php/jtla/article/download/1650/1492
