---
title: 'RSMTool: collection of tools building and evaluating automated scoring models'
tags:
  - educational applications
  - scoring models
  - statistics
  - visualization
authors:
 - name: Nitin Madnani
   orcid: 0000-0001-9354-6851
   affiliation: Educational Testing Service
 - name: Anastassia Loukina
   orcid:
   affiliation: Educational Testing Service
date: 23 June 2016
bibliography: bibliography.bib
---

# Summary

RSMTool is a collection of tools for researchers working on statistical models for automated scoring of written and spoken responses in an educational context. 

RSMTool takes as input a feature file with numeric, non-sparse features extracted from the responses and a human score and lets you try several different machine learning algorithms to try and predict the human score from the features. The primary output of RSMTool is a comprehensive, customizable HTML statistical report that contains feature descriptives, subgroup analyses, model statistics, as well as several different evaluation measures illustrating model efficacy. The various numbers and figures in the report are highlighted based on whether they exceed or fall short of the recommendations laid out by Williamson et al. [@Williamson2012]. The structure of RSMTool makes it easy for researchers to add new analyses without making any changes to the core code structure thus allowing for a wide range of psychometric evaluations. 

The tool is written in Python and works on all platforms (Windows/Linux/Mac OS X). The source code is available from Github [@github_rsmtool]

# References
