## Introduction

RSMTool is a python package for facilitating research on building and evaluating scoring models (SMs) for automated scoring engines by allowing the integration of educational measurement practices with the automated scoring and model building process. 

Specifically, RSMTool takes a feature file with numeric, non-sparse features and a human score as input and lets you try several different regression models to try and predict the human score from the features. The primary output of RSMTool is a comprehensive, customizable HTML statistical report that contains feature descriptives, subgroup analyses, model statistics, as well as several different evaluation measures illustrating model efficacy. The various numbers and figures in the report are highlighted based on whether they exceed or fall short of the recommendations laid out by Williamson et al. (2012). However, these can be easily customized if the user wishes to use different set of recommendations.

Finally, since the report is based on IPython notebooks, it can be easily customized. In addition, RSMTool explicitly provides support for adding custom notebooks to the report. [Here's](http://bit.ly/rsmtool) an example RSMTool report for a simple scoring system built to automatically score the responses from the [2012 Kaggle Automated Student Assessment Prize competition](https://www.kaggle.com/c/asap-aes). 


RSMTool provides the following main scripts:

* `rsmtool` - the tool for training and evaluating scoring models. 

* `rsmeval` - the tool for evaluating predictions obtained from other systems. 

* `rsmpredict` - the tool for generating new predictions based on an existing models. 

* `rsmcompare` -  for comparing two `rsmtool` runs.


*References*:
David M. Williamson, Xiaoming Xi, and F. Jay Breyer. 2012. A Framework for Evaluation and Use of Automated Scoring. Educational Measurement: Issues and Practice, 31(1):2–13.

## Installation

If you want to use RSMTool on your own machine, either as a user or a developer, follow the appropriate instructions below. Note that RSMTool only works with Python 3.4 and higher. 

### For users

Currently, the best way to install RSMTool is by using the `conda` package manager. If you have the `conda` package manager already installed, you can skip straight to Step 2. 

1. To install the `conda` package manager, follow the instructions on [this page](http://conda.pydata.org/docs/install/quick.html).  

2. Create a new conda environment (say, `rsmtool`) and install the `rsmtool` conda package by running `conda create -n rsmtool -c desilinguist python=3.4 rsmtool`.

3. Activate this conda environment by running `source activate rsmtool`. You should now have all of the four tools above in your path.

4. From now on, you will need to activate this conda environment whenever you want to use RSMTool. This will ensure that the packages required by `rsmtool` will only be used when you want to run `rsmtool` experiments and will not affect other projects. 

### For developers

The instructions below are only if you are developing new features or functionality for RSMTool.

1. Pull the latest version of rsmtool from github and switch to the develop branch. 

2. If you already have the `conda` package manager installed, skip to the next step. If you do not, follow the instructions on [this page](http://conda.pydata.org/docs/install/quick.html) to install `conda`. 

3. Create a new conda environment (say, `rsmtool`) and install the packages specified in the `conda_requirements.txt` file by running `conda create -n rsmtool -c desilinguist --file conda_requirements.txt`. Use `conda_requirements_windows.txt` if you are on Windows. The two conda requirements file will be consolidated with the next version.

4. Activate the environment using `source activate rsmtool` (use `activate rsmtool` if you are on Windows).

5. Run `pip install -e .` to install rsmtool into the environment in editable mode which is what we need for development.  

6. Run `nosetests -v tests` to run the tests.  

## Available documentation

## Usage documentation for main scripts

* [rsmtool](doc/rsmtool.md) 

* [rsmeval](doc/rsmeval.md)  

* [rsmpredict](doc/rsmpredict.md) 

* [rsmcompare](doc/rsmcompare.md) 

## Description of configuration files

* [RSMTool configuration file](doc/config_file.md) - main configuration file for `rsmtool`

* [RSMEval configuration file](doc/config_file_eval.md) - main configuration file for `rsmeval`

* [RSMPredict configuration file](doc/config_file_eval.md) - main configuration file for `rsmpredict`

* [RSMCompare configuration file](doc/config_file_eval.md) - main configuration file for `rsmcompare`

* [Feature file](doc/feature_file.md) - feature file

## Lists of available options

* [Available models](doc/available_models.md) - list of models available to `rsmtool`

* [Report sections](doc/report_sections.md) - list of sections that can be included into the report

## Description of outputs

* [Output CSV files](doc/output_csv.md) - .csv files generated by `rsmtool` and `rsmeval`

## Documentation for developers

* [New notebooks](doc/new_notebooks.md) - the variables and data frames available for use in custom report sections.

* [Release process](doc/release_process.md) - the description of the release process for new versions.

