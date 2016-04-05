# Description of output files created by RSMTool and RSMEval

`rsmtool` and `rsmeval` create a directory called 'output' which contains various `.csv` files generated during feature preprocessing, model building and evaluation. All files begin with the `experiment_id`. 

The names for columns such as `id_column` or `length_column` are standardized in the output files using the default values specified in `config_file.md` and `config_file_eval.md`. For example, if you specified `sc2` as `train_label_column` this column will still appear as `sc1` in the output files.

## Data files [rsmtool]

By default RSMTool filters out non-numeric feature values and non-numeric/zero human scores from both training and evaluation set. Zero scores can be kept by setting the `exclude_zero_scores` to `false`.

* `*_train_preprocessed_features.csv`/`*_test_preprocessed_features.csv` - feature values for training/evaluation set after applying pre-processing. These files only include the rows used to train/evaluate the model as above. For models with feature selection these files include the features that were used for feature selection. 

* `*_train_features.csv`/`*_test_features.csv` - raw feature values for the training/evaluation set. This file only includes the rows that were used for training/evaluating the models after filtering. For models with feature selection these files include the features that were used for feature selection. 

The following files are only saved if not empty:

* `*_train_responses_with_excluded_flags.csv`/`*_test_responses_with_excluded_flags.csv` - all columns (raw feature values, metadata and all other columns) for the responses that were filtered out based on conditions specified in `flag_column`.  Note that if you used the default names such as `sc1` or `length` for any columns not used for the analysis, these columns will be included into these files but their names will be changed to `##name##` (e.g. `##sc1##`).

* `*_train_excluded_responses.csv`/`*_test_excluded_responses.csv` - all columns including raw feature values for all responses that were filtered out because of feature values or scores. For models with feature selection these files include the features that were used for feature selection. 

* `*_train_metadata.csv`/`*_test_metadata.csv` - metadata (id_column,  subgroups if requested) only for the data used to train/evaluate the model as above. 

* `*_train_other_columns.csv`/`*_test_other_columns.csv` - the columns from the original file other than the ones included into `_feature.csv` and `_metadata.csv`. This file only includes the rows that were used for training/evaluating the models after filtering. Note that if you used the default names such as `sc1` or `length` for any columns not used for the analysis, these columns will be included into `other_columns` but their name will be changed to `##name##` (e.g. `##sc1##`)

* `*_train_response_lengths.csv` - If `length_column` is specified, then this CSV file contains the values from that column for the training data under a column called `length` and also contains the `spkitemid` column.

* `*_test_human_scores.csv` - If `second_human_score_colunmn` is specfied, then this CSV file contains the values from that column for the test data under a column called `sc2` and also contains the `spkitemid` and the `sc1` column. The table only includes the rows that were used for training/evaluating the model after filtering. Note that if `exclude_zero_scores`  was set to `True` (the default value), all zero scores in the `second_human_score_column` will be replaced by `nan`.

## Data composition analyses [rsmtool, rsmeval]

* `*_data_composition.csv` - total N responses in training and evaluation set and the number of overlapping responses. If applicable, the table will also include the number of different subgroups for each set.

* `*_train_excluded_composition.csv`/`*_test_excluded_composition.csv` - the analysis of excluded responses for training and evaluation set. 

* `*_train_missing_feature_values.csv` - total number of non-numeric feature values for each feature and (if available) the distribution of `numwds` in responses with non-numeric feature values. All counts in this table only include responses with numeric human score. 

### Subgroup analyses (if requested)

* `*_data_composition_by_SUBGROUP.csv` - total number of responses in each category of SUBGROUP in training and evaluation set. 

## Feature analysis files [rsmtool]

The results in these files are computed on the training set. 

* `*_feature_descriptives.csv` - main descriptive statistics (mean,std. dev., correlation with human score etc.) for all features used in the final model. These are computed on raw feature values before pre-processing. 

* `*_feature_descriptivesExtra.csv` - percentiles, mild and extreme outliers for for all features used in the final model. These are computed on raw feature values before pre-processing. 

* `*_feature_outliers.csv` - number and percentage of cases truncated during feature pre-processing for each feature included in the final model. 

* `*_cors_orig.csv`/`*_cors_processed.csv`  - correlation matrix for raw/preprocessed feature values and human score ('sc1')

* `*_margcor_score_all_data.csv` - marginal correlations between each feature and human score.

* `*_pcor_score_all_data.csv` - partial correlations between each feature and human score after controlling for all other features. 

* `*_pcor_score_no_length_all_data.csv` - partial correlations between each feature and human score after controlling for response length, if `length_column` was specified in the config file.

* `*_margcor_length_all_data.csv` - marginal correlations between each feature and response length, if `length_column` was specified in the config file.

* `*_pcor_length_all_data.csv` - partial correlations between each feature and response length after controlling for all other features, if `length_column` was specified in the config file.

* `*_pca.csv` - the results of the Principal Component Analysis using pre-processed feature values and singular value decomposition. 

* `*_pcavar.csv` - eigenvalues and variance explained by each component. 

### Subgroup analyses (if requested)

* `*_margcor_score_by_SUBGROUP.csv` - marginal correlations between each feature and human score computed separately for each subgroup. 

* `*_pcor_score_by_SUBGROUP.csv` - partial correlations between each feature and human score after controlling for all other features computed separately for each subgroup. 

* `*_pcor_score_no_length_by_SUBGROUP.csv` - partial correlations between each feature and human score after controlling for response length, if `length_column` was specified in the config file.

* `*_margcor_length_by_SUBGROUP.csv` - marginal correlations between each feature and response length computed separately for each subgroup, if `length_column` was specified in the config file. 

* `*_pcor_length_by_SUBGROUP.csv` - partial correlations between each feature and response length after controlling for all other features computed separately for each subgroup, if `length_column` was specified in the config file. 

## Model files [rsmtool]

* `*_feature.csv` - pre-processing parameters for all features used in the model. 

* `*_coefficients.csv` - model coefficients and intercept (linear models only)

* `*_coefficients_scaled.csv` - scaled model coefficients and intercept (linear models only) for generating scaled scores. RSMTool generates scaled scores by scaling predictions. It is also possible to achieve the same result by scaling coefficients.

* `*_betas.csv`- standardized and relative coefficients linear models only)

* `*_.model` - SKLL object with fitted model (before scaling the coeffcients)

* `*_.ols` - a pickled object containing a fitted model of type `pandas.stats.ols.OLS` (for built-in linear models, excluding `LassoFixedLabmda` and `PositiveLassoCV`)

* `*_ols_summary.txt` - a text file containing a summary of the above model (for built-in linear models, excluding `LassoFixedLabmda` and `PositiveLassoCV`)

* `*_postprocessing_params.csv` - the parameters for trimming and scaling predicted scores for new predictions. 


## Predictions [rsmtool, rsmeval]

* `*_pred_processed.csv` - predicted scores for the evaluation set: this file includes raw scores as well as different types of post-processed scores. 

* `*_pred_train.csv` - predicted scores for the training set (rsmtool only)


## Evaluation files [rsmtool, rsmeval]

The results in these files are computed on evaluation set. 

* `*_eval.csv` -  descriptives for predicted and human scores (mean, std.dev etc.) and  association metrics (correlation, quadartic weighted kappa, SMD etc.) for all types of scores (raw, scaled, trimmed, rounded).

* `*_eval_short.csv` -  a shortened version of `eval.csv` that contains specific descriptives for predicted and human scores (mean, std.dev etc.) and  association metrics (correlation, quadartic weighted kappa, SMD etc.) for specific score types chosen based on recommendations by Williamson (2012). Specifically, the following columns are included (the `raw` or `scale` version is chosen depending on the value of the `use_scaled_predictions`).

    ```
    - h_mean
    - h_sd
    - corr [raw/scale trim]
    - sys_mean [raw/scale trim]
    - sys_sd [raw/scale trim]
    - SMD (raw/scale trim)
    - adj_agr (raw/scale trim_round)
    - exact_agr [raw/scale trim_round]
    - kappa [raw/scale trim_round]
    - wtkappa [raw/scale trim_round]
    - sys_mean [raw/scale trim_round]
    - sys_sd [raw/scale trim_round]
    - SMD (raw/scale trim_round)
    ```

* `*_score_dist.csv` - distribution of rounded predicted and human scores using raw/scaled scores as specified in the configuration file. 

* `*_confMatrix.csv` - confusion matrix between rounded predicted and human scores using raw/scaled scores as specified in the configuration file.

### Human-human consistency (if requested) [rsmtool, rsmeval]

* `*_consistency.csv` - descriptives for both human raters and the agreement between the two human raters. 

* `*_degradataion.csv` - degradation between human-human agreement and machine-human agreement for all association metrics and all types of scores. 

### Subgroup analyses (if requested) [rsmtool, rsmeval]

* `*_eval_by_SUBGROUP.csv` - same information as in `*_eval.csv` computed separately for each subgroup. 

