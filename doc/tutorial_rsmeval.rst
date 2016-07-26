.. _tutorial_rsmeval:

Tutorial
""""""""

For this tutorial, you first need to :ref:`install RSMTool <install>`.

Workflow
~~~~~~~~

``rsmeval`` is designed for evaluating existing machine scores. Once you have the scores computed for all the responses in your data, the next steps are fairly straightforward:

1. Create a ``.csv`` file containing the computed system scores and the human scores you want to compare against.
2. Create an :ref:`experiment configuration file <config_file_rsmeval>` describing the evaluation experiment you would like to run.
3.  Run that configuration file with :ref:`rsmeval <usage_rsmeval>` and generate the experiment HTML report as well as the :ref:`intermediate CSV files <intermediate_files_rsmeval>`.
4. Examine the HTML report to check various aspects of model performance.

Note that the above workflow does not use any customization features , e.g., :ref:`choosing which sections to include in the report <general_sections_rsmeval>` or :ref:`adding custom analyses sections <custom_notebooks>` etc. However, we will stick with this workflow for our tutorial since it is likely to be the most common use case.

ASAP Example
~~~~~~~~~~~~
We are going to use the same example from 2012 Kaggle competition on automated essay scoring that we used for the :ref:`rsmtool tutorial <tutorial>`.

Generate scores
~~~~~~~~~~~~~~~
``rsmeval`` is designed for researchers who have developed their own scoring engine for generating scores and would like to produce an evaluation report for those scores.  For this tutorial, we will use the scores we generated for the ASAP2 evaluation set using :ref:`rsmtool tutorial <tutorial>`.

Create a configuration file
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The next step is to create an :ref:`experiment configuration file <config_file_rsmeval>` in ``.json`` format.

.. _asap_config_rsemeval:

.. literalinclude:: ../examples/rsmeval/config_rsmeval.json
    :language: javascript
    :linenos:

Let's take a look at the options in our configuration file.

- **Line 2**: We define an experiment ID.
- **Line 3**: We also provide a description which will be included in the experiment report.
- **Line 4**: We list the path to the ``.csv`` with the predicted and human scores.
- **Line 5**: This field indicates that the system scores in our ``.csv`` file are located in a column named ``system``.
- **Line 6**: This field indicates that the human (reference) scores in our ``.csv`` file are located in a column named ``human``.
- **Line 7**: This field indicates that the unique IDs for the responses in the ``.csv`` file are located in columns named ``ID``.
- **Lines 8-9**: These fields indicate that the lowest score on the scoring scale is a 1 and the highest score is a 6. This information is usually part of the rubric used by human graders.
- **Line 10**: This field indicates that scores from a second set of human graders are also available (useful for comparing the agreement between human-machine scores to the agreement between two sets of humans) and are located in the ``human2`` column in the ``.csv`` file.
- **Line 11**: This field indicates that the provided machine scores are already re-scaled to match the distribution of human scores. ``rsmeval`` itself will not perform any scaling and the report will refer to these as ``scaled`` scores.

Documentation for all of the available configuration options is available :ref:`here <config_file_rsmeval>`.

Run the experiment
~~~~~~~~~~~~~~~~~~
Now that we have our scores in ``.csv`` format and our configuration file in ``.json`` format, we can use the :ref:`rsmeval <usage_rsmeval>` command-line script to run our evaluation experiment.

.. code-block:: bash

    $ cd examples/rsmeval
    $ rsmeval config_rsmeval.json

This should produce output like::

    Output directory: /Users/nmadnani/work/rsmtool/examples/rsmeval
    Assuming given system predictions are already scaled and will be used as such.
     predictions: /Users/nmadnani/work/rsmtool/examples/rsmeval/ASAP2_scores.csv
    Processing predictions
    Saving pre-processed predictions and the metadata to disk
    Running analyses on predictions
    Starting report generation
    Merging sections
    Exporting HTML
    Executing notebook with kernel: python3


Once the run finishes, you will see the ``output``, ``figure``, and ``report`` sub-directories in the current directory. Each of these directories contain :ref:`useful information <output_dirs_rsmeval>` but we are specifically interested in the ``report/ASAP2_evaluation_report.html`` file, which is the final evaluation report.

Examine the report
~~~~~~~~~~~~~~~~~~
Our experiment report contains all the information we would need to evaluate the provided system scores against the human scores. It includes:

1. The distributions for the human versus the system scores.
2. Several different metrics indicating how well the machine's scores agree with the humans'.
3. Information about human-human agreement and the difference between human-human and human-system agreement.

... and :ref:`much more <general_sections_rsmeval>`.
