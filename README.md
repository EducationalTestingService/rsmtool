[![Circle CI](https://circleci.com/gh/EducationalTestingService/rsmtool/tree/master.svg?style=shield)](https://circleci.com/gh/EducationalTestingService/rsmtool/tree/master)
[![Coverage Status](https://coveralls.io/repos/github/EducationalTestingService/rsmtool/badge.svg?branch=feature%2Fadd-test-coverage)](https://coveralls.io/github/EducationalTestingService/rsmtool?branch=feature%2Fadd-test-coverage)
[![DOI](https://zenodo.org/badge/22127/EducationalTestingService/rsmtool.svg)](https://zenodo.org/badge/latestdoi/22127/EducationalTestingService/rsmtool)
[![GitHub license](https://img.shields.io/badge/license-Apache%202-blue.svg)](https://raw.githubusercontent.com/EducationalTestingService/rsmtool/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/rsmtool/badge/?version=latest)](http://rsmtool.readthedocs.io/en/latest/?badge=latest)

## RSMTool

Automated scoring of written and spoken test responses is a growing field in educational natural language processing. Automated scoring engines employ machine learning models to predict scores for such responses based on features extracted from the text/audio of these responses. Examples of automated scoring engines include [Project Essay Grade](http://pegwriting.com/about) for written responses and [SpeechRater](https://www.ets.org/research/topics/as_nlp/speech/) for spoken responses.

RSMTool is a python package which automates and combines in a single pipeline multiple analyses that are commonly conducted when building and evaluating such scoring models.  The output of RSMTool is a comprehensive, customizable HTML statistical report that contains the output of these multiple analyses. While RSMTool does make it really simple to run a set of standard analyses using a single command, it is also fully customizable and allows users to easily exclude unneeded analyses, modify the default analyses, and even include custom analyses in the report.

We expect the primary users of RSMTool to be researchers working on developing new automated scoring engines or on improving existing ones. Note that RSMTool is not a scoring engine by itself but rather a tool for building and evaluating machine learning models that may be used in such engines. 

For installation and usage, please see the official [documentation](http://rsmtool.readthedocs.io). 

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



