Tutorial
========

Before doing anything below, you'll want to :ref:`install RSMTool <install>`.

Workflow
--------

In general, there are four steps to using RSMTool:

1.  Generate features for your training as well as your evaluation data using your NLP/Speech pipline in ``.csv`` format.
2. Create an :ref:`experiment configuration file <config_file_rsmtool>` describe the modeling experiment you would like to run.
3.  Run that configuration file with :ref:`rsmtool <usage_rsmtool>` and generate the experiment HTML report as well as the :ref:`intermediate CSV files <intermediate_files_rsmtool>`.
4. Examine HTML report for model efficacy.

Note that the above workflow does not use the customization features of RSMTool, e.g., :ref:`choosing which sections to include in the report <general_sections_rsmtool>` or :ref:`adding custom analyses sections <custom_notebooks>` etc. However, we will stick with this workflow for our tutorial since it is likely to be the most common use case.

ASAP Example
------------

Let's see how we can apply this basic RSMTool workflow to a simple example based on a 2012 Kaggle competition on automated essay scoring called the `Automated Student Assessment Prize (ASAP) contest <https://www.kaggle.com/c/asap-aes>`_. As part of that contest, responses to 8 different essay questions written by students in grades 6-10 were provided. The responses were scored by humans and given a holistic score indicating the English proficiency of the student. The goal of the contest was to build an automated scoring model with the highest accuracy on held-out evaluation data.

For our tutorial, we will use one of the questions from this data to illustrate how to use RSMTool to train a model and evaluate its performance.

Extract features
^^^^^^^^^^^^^^^^
Kaggle provides the actual text of the responses along with the human scores in a ``.tsv`` file. The first step, as with any automated scoring approach, is to extract features from the written (or spoken) responses that might be useful in predicting the score we are interested in. For example, some features that might be useful for predicting the English proficiency of a response are:

- Does the response contain any grammatical errors and if so, how many and of what kind since some grammatical errors are more serious than others.
- Is the response is well organized, e.g., whether there is a clear thesis statement and a conclusion, etc.
- Is the response coherent, i.e., do the ideas expressed in the various paragraphs well connected?
- Does the response contain any spelling errors?

For more examples of what features might look like, we refer the reader to this paper by Attali & Burstein [#]_. For our ASAP2 data, we extract the following four features:

- ``GRAMMAR``: indicates how grammatically fluent the responses is.
- ``MECHANICS``: indicates how free of mechanical errors (e.g., spelling errors) the response is.
- ``DISCOURSE``: indicates whether the discourse transitions in the response make sense.
- ``ORGANIZATION``: indicates whether the response is organized well.

For the purpose of this tutorial, it is not important to describe *exactly* how those features were extracted. You would likely use your own NLP/Speech processing pipeline to extract your own features.

Create a configuration file
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The next step is to create an :ref:`RSMTool configuration file <config_file_rsmtool>` in ``.json`` format.

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
- **Lines 10-11**: We indicate that the lowest score on the scoring scale is a 1 and the highest score is a 6. This information is usually part of the rubric used by human graders.
- **Line 12**: We indicate that the unique IDs for the responses in the two ``.csv`` files are located in a column named ``ID``.


.. rubric:: References

.. [#] Attali, Y., & Burstein, J. (2006). Automated essay scoring with e-rater® V.2. Journal of Technology, Learning, and Assessment, 4(3). https://ejournals.bc.edu/ojs/index.php/jtla/article/download/1650/1492
