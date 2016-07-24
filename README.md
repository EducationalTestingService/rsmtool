[![Circle CI](https://circleci.com/gh/EducationalTestingService/rsmtool/tree/master.svg?style=shield)](https://circleci.com/gh/EducationalTestingService/rsmtool/tree/master)
[![Coverage Status](https://coveralls.io/repos/github/EducationalTestingService/rsmtool/badge.svg?branch=feature%2Fadd-test-coverage)](https://coveralls.io/github/EducationalTestingService/rsmtool?branch=feature%2Fadd-test-coverage)
[![DOI](https://zenodo.org/badge/22127/EducationalTestingService/rsmtool.svg)](https://zenodo.org/badge/latestdoi/22127/EducationalTestingService/rsmtool)
[![GitHub license](https://img.shields.io/badge/license-Apache%202-blue.svg)](https://raw.githubusercontent.com/EducationalTestingService/rsmtool/master/LICENSE)

## RSMTool

Automated scoring of written and spoken test responses is a growing field in educational natural language processing. Automated scoring engines employ machine learning models to predict scores for such responses based on features extracted from the text/audio of these responses. Examples of automated scoring engines include [Project Essay Grade](http://pegwriting.com/about) for written responses and [SpeechRater](https://www.ets.org/research/topics/as_nlp/speech/) for spoken responses.

RSMTool is a python package which automates and combines in a single pipeline multiple analyses that are commonly conducted when building and evaluating such scoring models.  The output of RSMTool is a comprehensive, customizable HTML statistical report that contains the multiple analyses required for a comprehensive evaluation of an automated scoring model. While RSMTool does makes it really simple to run a set of standard analyses using a single command, it is also fully customizable and allows users to easily exclude unneeded analyses, modify the default analyses, and even include custom analyses in the report.

We expect the primary users of RSMTool to be researchers working on developing new automated scoring engines or on improving existing ones. Note that RSMTool is not a scoring engine by itself but rather a tool for building and evaluating machine learning models that may be used in such engines. 

For details on usage and installation, please see the official [documentation](http://rsmtool.readthedocs.io). 

## Requirements

- Python 3.4 or higher
- `numpy`
- `scipy`
- `scikit-learn`
- `statsmodels`
- `skll`
- `pandas`
- `ipython`
- `jupyter`
- `notebook`
- `seaborn`
- `setuptools`

## Contributing

Contributions to RSMTool are very welcome. Please refer to the [documentation](http://rsmtool.readthedocs.io/en/latest/contributing.html) for how to get started on developing new features or functionality for RSMTool.


## Changelog
See [GitHub Releases](https://github.com/EducationalTestingService/rsmtool/releases).

**Use case 2**: A researcher *has* a proprietary automated scoring system (using a machine learning model that is not available in RSMTool) for grading short responses that extracts the features and computes the score. He wants to evaluate the system performance using analyses recommended by the educational measurement community that are not always available in standard machine learning packages. In addition, he also wants to conduct additional analyses to evaluate system fairness and compare whether his system agrees with the human grader as well as a second human grader would.

He *uses* RSMTool to set up a customized evaluation report using a combination of existing and custom sections and then runs RSMEval to quickly produce a new report for each version of his system.


RSMTool provides the following main scripts. The documentation for each of these tools is provided separately (see below). 

* `rsmtool` - the tool for training and evaluating scoring models as described in Use case 1

* `rsmeval` - the tool for evaluating predictions obtained from other systems as described in use case 2. 

* `rsmpredict` - the tool for generating new predictions based on an existing models. For researchers who are already using `rsmtool` to build scoring model, this tool allows generating predictions for new data using an existing model. 

* `rsmcompare` -  for comparing two `rsmtool` runs. This tool compares two models trained using `rsmtool`. It is most commonly used to compare model performance after adding new features.


## Example

You can try out RSMTool as follows:

1. Go to the `example/rsmtool` folder. This folder contains the training and test set features for a simple scoring system built to automatically score the responses from the [2012 Kaggle Automated Student Assessment Prize competition](https://www.kaggle.com/c/asap-aes). 
2. Make sure to activate the conda environment where you installed rsmtool (e.g., `source activate rsmtool`)
3. We first try the whole pipeline by running RSMTool: `rsmtool config.json`
4. Since no output directory was specfied, `rsmtool` will create the three output folders in the current directory: `figure`, `output`, and `report`. You can examine the HTML report `report/ASAP2_report.html`. It should look like [this](https://s3.amazonaws.com/sample-rsmtool-report/ASAP2_report.html).
5. Now we will use `rsmpredict` to re-generate the scores for the test set without re-training the model. We will store these new predictions in `predictions.csv`: go to `../rsmpredict` and run `rsmpredict config_rsmpredict.json predictions.csv`. The tool will create a new file called `predictions.csv` which contains the predictions. 
6. We will use `rsmeval` to evaluate these new predictions: go to `../rsmeval` and run `rsmeval config_rsmeval.json` .`rsmeval` will create the three output folders in the current directory: `figure`, `output`, and `report`. 
7. Finally, we will compare the two sets of predictions by running `RSMCompare`: go to `../rsmcompare` and run `rsmcompare config_rsmcompare.json`. The tool will create an `.html` report in the current directory since no output directory was specified. This report compare the analyses generated at steps (4) and (5). Note that since for `rsmeval` we used existing predictions, no information is available about feature distributions of model parameters. 

## Contributing

Contributions to RSMTool are very welcome. You can use the instructions below to get started on developing new features or functionality for RSMTool.

1. Pull the latest version of rsmtool from github and switch to the `master` branch. 

2. If you already have the `conda` package manager installed, skip to the next step. If you do not, follow the instructions on [this page](http://conda.pydata.org/docs/install/quick.html) to install `conda`. 

3. Create a new conda environment (say, `rsmtool`) and install the packages specified in the `conda_requirements.txt` file by running `conda create -n rsmtool -c desilinguist --file conda_requirements.txt`. Use `conda_requirements_windows.txt` if you are on Windows. There are two versions because RSMTool currently does not use MKL on non-Windows platforms.

4. Activate the environment using `source activate rsmtool` (use `activate rsmtool` if you are on Windows).

5. Run `pip install -e .` to install rsmtool into the environment in editable mode which is what we need for development.  

6. Run `nosetests -v tests` to run the tests. 

## Available documentation

### Usage documentation for main scripts

* [rsmtool](doc/rsmtool.md) 

* [rsmeval](doc/rsmeval.md)  

* [rsmpredict](doc/rsmpredict.md) 

* [rsmcompare](doc/rsmcompare.md) 

### Description of configuration files

* [RSMTool configuration file](doc/config_file.md) - main configuration file for `rsmtool`

* [RSMEval configuration file](doc/config_file_eval.md) - main configuration file for `rsmeval`

* [RSMPredict configuration file](doc/config_file_eval.md) - main configuration file for `rsmpredict`

* [RSMCompare configuration file](doc/config_file_eval.md) - main configuration file for `rsmcompare`

* [Feature file](doc/feature_file.md) - feature file

### Lists of available options

* [Available models](doc/available_models.md) - list of models available to `rsmtool`

* [Report sections](doc/report_sections.md) - list of sections that can be included into the report

## Description of outputs

* [Output CSV files](doc/output_csv.md) - .csv files generated by `rsmtool` and `rsmeval`

### Documentation for developers

* [New notebooks](doc/new_notebooks.md) - the variables and data frames available for use in custom report sections.

* [Release process](doc/release_process.md) - the description of the release process for new versions.

