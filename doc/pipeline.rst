Overview of RSMTool Pipeline
============================

The following figure gives an overview of the RSMTool pipeline:

.. image:: pipeline.png
   :alt: RSMTool processing pipeline
   :align: center

As its primary input, RSMTool takes a CSV feature file with numeric, non-sparse features and a human score as input and lets you try several different regression models (including Ridge, SVR, AdaBoost, and Random Forests) to try and predict the human score from the features.

The primary output of RSMTool is a comprehensive, customizable HTML statistical report that contains the multiple analyses required for a comprehensive evaluation of an automated scoring model including feature descriptives, subgroup comparisons, model statistics, as well as several different evaluation measures illustrating model efficacy. Details about these various analyses are provided in a separate `technical paper <https://github.com/EducationalTestingService/rsmtool/raw/master/doc/rsmtool.pdf>`_.

In addition to the HTML report, RSMTool also saves the intermediate outputs of all of the performed analyses as CSV files.



