# General sections that can be included into the report

There are three types of sections that can be included into the report:

1. The *general sections* are supplied as a part of `rsmtool` and applicable to most automated scoring systems and will always be included by default unless the user uses the field `general_sections` to specify only a subset of all sections. 

2. The *special sections* are available only to ETS users as a part of `rsmextra` package and are applicable to only some engines and/or data. These sections will only be included if requested by the user using the `special_sections` field. See the documentation to `rsmextra` for the list of available sections.

3. All users can create their own *custom sections* that can be included into the report. See [config_file.md](config_file.md)/[config_file_eval.md](config_file.md) and [new_notebooks.md](config_file.md) for more information about creating such sections. Note that *custom sections* are not part of the git repository for RSMTool and are not included into new releases. 

Please note that different sections are available to different tools. The list of sections below specifies which *general sections* can be used with which tool.  Please see the documentation for `rsmextra` for the list of *special sections*. 

## Data description [rsmtool, rsmeval]

* `data_description`: total number of responses in training and evaluation set. If any responses have been excluded due to non-numeric features/scores or flag columns, this section will show further analysis of these responses.

* `data_description_by_group`: total number of responses in each category in training and evaluation set for each of the subgroups specified in the configuration file. This section only covers the responses used to train/evaluate the model.

## Descriptive analyses on the training set [rsmtool, rsmcompare]

* `feature_descriptives`: main descriptive statistics for raw values of all features included into the model. (1) a table showing mean, standard deviation, min, max, correlation with human score etc; (2) A table with percentiles and outliers; (3) a barplot showing he number of truncated outliers for each feature included into the model.

* `features_by_group`: boxplots showing the distribution of raw feature values by each of the subgroups specified in the configuration file. 

* `preprocessed_features`: evaluation of preprocessed features: (1) historgrams showing the distributions of preprocessed features values; (2) the correlation matrix between all features and the human score; and (3) the barplot showing marginal and partial correlations between all features and the human score and, optionally, response length if `length_column` is specified in the config file. 

* `preprocessed_features_by_group`: marginal and partial correlations (after controlling for all other features) between each feature and human score for each subgroup (rsmcompare only).

* `pca`: the results of the principal component analysis (1) Principal components (2) Variance (3) Scree plot. 

## Model parameters [rsmtool, rsmcompare]

* `model`: model parameters. For linear models this section also includes the plots showing standardized and relative coefficients and the model diagnostic plots. 

## Model evaluation on the evaluation set [rsmtool, rsmeval, rsmcompare]

* `evaluation`: standard evaluations as recommended by Williamson (2012). (1) A table showing all standard system-human association matrics for all types of scores; (2) Confusion matrix; (3) A barplot showing score distributions.
Note: for `rsmcompare` score distributions are currently shown in a separate notebook called `score_distributions`.

* `evaluation by group`: barplots showing the main evaluation metrics (SMD, quadratic weighted kappa, and correlation) by each of the subgroups specified in the configuration file. 


## Auxiliary sections [rsmtool, rsmeval, rsmcompare]

* `sysinfo`: Information about the versions of the Python packages used while generating the report. 

* `notes` (rsmcompare only): Notes explaining the terminology used in rsmcompare. 



