[![Circle CI](https://circleci.com/gh/EducationalTestingService/rsmtool/tree/master.svg?style=shield)](https://circleci.com/gh/EducationalTestingService/rsmtool/tree/master)
[![Coverage Status](https://coveralls.io/repos/github/EducationalTestingService/rsmtool/badge.svg?branch=master)](https://coveralls.io/github/EducationalTestingService/rsmtool?branch=master)
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.58851.svg)](http://dx.doi.org/10.5281/zenodo.58851)
[![status](http://joss.theoj.org/papers/fbc649c17d45074d92ac21084aaa6209/status.svg)](http://joss.theoj.org/papers/fbc649c17d45074d92ac21084aaa6209)
[![GitHub license](https://img.shields.io/badge/license-Apache%202-blue.svg)](https://raw.githubusercontent.com/EducationalTestingService/rsmtool/master/LICENSE)
[![Documentation Status](https://readthedocs.org/projects/rsmtool/badge/?version=latest)](http://rsmtool.readthedocs.io/en/latest/?badge=latest)

## RSMTool

Automated scoring of written and spoken test responses is a growing field in educational natural language processing. Automated scoring engines employ machine learning models to predict scores for such responses based on features extracted from the text/audio of these responses. Examples of automated scoring engines include [Project Essay Grade](http://pegwriting.com/about) for written responses and [SpeechRater](https://www.ets.org/research/topics/as_nlp/speech/) for spoken responses.

RSMTool is a python package which automates and combines in a single pipeline multiple analyses that are commonly conducted when building and evaluating such scoring models.  The output of RSMTool is a comprehensive, customizable HTML statistical report that contains the output of these multiple analyses. While RSMTool does make it really simple to run a set of standard analyses using a single command, it is also fully customizable and allows users to easily exclude unneeded analyses, modify the default analyses, and even include custom analyses in the report.

We expect the primary users of RSMTool to be researchers working on developing new automated scoring engines or on improving existing ones. Note that RSMTool is not a scoring engine by itself but rather a tool for building and evaluating machine learning models that may be used in such engines. 

For installation and usage, please see the official [documentation](http://rsmtool.readthedocs.io). 

## Requirements

- Python 3.6
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

## Citing
If you are using RSMTool in your work, you can cite it as follows:

### MLA
Madnani, Nitin and Loukina, Anastassia. "RSMTool: A Collection of Tools for Building and Evaluating Automated Scoring Models". Journal of Open Source Software 1(3), 2016.

### BibTex

```bib
@article{MadnaniLoukina2016,
  doi = {10.21105/joss.00033},
  url = {http://dx.doi.org/10.21105/joss.00033},
  year  = {2016},
  month = {jul},
  publisher = {The Open Journal},
  volume = {1},
  number = {3},
  author = {Nitin Madnani and Anastassia Loukina},
  title = {{RSMTool}: A Collection of Tools for Building and Evaluating Automated Scoring Models},
  journal = {{Journal of Open Source Software}}
}
```

## Changelog
See [GitHub Releases](https://github.com/EducationalTestingService/rsmtool/releases).

