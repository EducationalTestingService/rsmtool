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

Let's see how we can apply this basic RSMTool workflow to a simple example based on a 2012 Kaggle competition on automated essay scoring called the Automated Student Assessment Prize (ASAP) content. As part of that contest, human-scored responses to 8 different essay questions written by students in grades 6-10 were provided. The goal of the contest was to build an automated scoring model that had the highest accuracy when compared to the human scores on held-out evaluation data.

For our tutorial, we will use one of the questions from this data to illustrate how to use RSMTool to train a model and evaluate its performance.

Extracting features
^^^^^^^^^^^^^^^^^^^
Kaggle provides the actual text of the responses along with the human scores in a ``.tsv`` file. The first step, as with any automated scoring approach, is to extract features that we think might be useful when to convert the written (or spoken) responses


