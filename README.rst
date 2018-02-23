Rater Scoring Modeling Tool
---------------------------

.. image:: https://img.shields.io/travis/EducationalTestingService/rsmtool/master.svg
   :target: https://travis-ci.org/EducationalTestingService/rsmtool
   :alt: Build status

.. image:: https://img.shields.io/coveralls/EducationalTestingService/rsmtool/master.svg
   :target: https://coveralls.io/r/EducationalTestingService/rsmtool
   :alt: Coverage status

.. image:: https://img.shields.io/conda/v/desilinguist/rsmtool.svg
   :target: https://anaconda.org/desilinguist/rsmtool
   :alt: Conda package for SKLL

.. image:: https://img.shields.io/pypi/pyversions/rsmtool.svg
   :target: https://pypi.org/project/rsmtool/
   :alt: Supported python versions for RSMTool

.. image:: https://img.shields.io/readthedocs/rsmtool.svg
   :target: https://rsmtool.readthedocs.io
   :alt: Docs

.. image:: https://img.shields.io/badge/DOI-10.21105%2Fjoss.00033-blue.svg
   :target: http://joss.theoj.org/papers/10.21105/joss.00033
   :alt: DOI for citing RSMTool

.. image:: https://img.shields.io/pypi/v/rsmtool.svg
   :target: https://pypi.org/project/rsmtool/
   :alt: Latest version on PyPI

.. image:: https://img.shields.io/pypi/l/rsmtool.svg
   :alt: License

Introduction
------------

Automated scoring of written and spoken test responses is a growing field in educational natural language processing. Automated scoring engines employ machine learning models to predict scores for such responses based on features extracted from the text/audio of these responses. Examples of automated scoring engines include `Project Essay Grade <http://pegwriting.com/about>`_ for written responses and `SpeechRater <https://www.ets.org/research/topics/as_nlp/speech/>`_ for spoken responses.

Rater Scoring Modeling Tool (RSMTool) is a python package which automates and combines in a single pipeline multiple analyses that are commonly conducted when building and evaluating such scoring models.  The output of RSMTool is a comprehensive, customizable HTML statistical report that contains the output of these multiple analyses. While RSMTool does make it really simple to run a set of standard analyses using a single command, it is also fully customizable and allows users to easily exclude unneeded analyses, modify the default analyses, and even include custom analyses in the report.

We expect the primary users of RSMTool to be researchers working on developing new automated scoring engines or on improving existing ones. Note that RSMTool is not a scoring engine by itself but rather a tool for building and evaluating machine learning models that may be used in such engines. 

For installation and usage, please see the `official documentation <http://rsmtool.readthedocs.io>`_. 

Requirements
------------

- Python 3.6
- ``numpy`
- ``scipy``
- ``scikit-learn``
- ``statsmodels``
- ``skll``
- ``pandas``
- ``ipython``
- ``jupyter``
- ``notebook``
- ``seaborn``
- ``setuptools``

Contributing
------------
Contributions to RSMTool are very welcome. Please refer to the `documentation <http://rsmtool.readthedocs.io/en/latest/contributing.html>`_ for how to get started on developing new features or functionality for RSMTool.

Citing
------
If you are using RSMTool in your work, you can cite it as follows:

MLA
===
Madnani, Nitin and Loukina, Anastassia. "RSMTool: A Collection of Tools for Building and Evaluating Automated Scoring Models". Journal of Open Source Software 1(3), 2016.

BibTex
======

.. code:: bib

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


Changelog
---------
See `GitHub Releases <https://github.com/EducationalTestingService/rsmtool/releases>`_.
