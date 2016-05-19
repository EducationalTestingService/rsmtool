# Description of RSMTool config file.

This file determines the overal structure of the experiment. The file must be in .json format. 

## Required fields:

`experiment_id`: ID for your experiment. This can be any combination of alphanumeric values and must not contain spaces.

`model` : machine learning model you want to use. See [available models](available_models.md) for the list of available learners. 

`train_file`: path to the training data in .csv format. Can be absolute or relative to the location of config file.

`test_file`: path to the evaluatin data in .csv format. Can be absolute or relative to the location of config file.

## Optional fields:

`description`: A brief description of the model. This will be printed in the report. The description can contain spaces and punctuation.
Default: None

### Column names

`id_column` - the name of the column containing the response IDs. 
Default: `spkitemid`

`train_label_column`: the label for human scores used to train the model. If this is set to 'fake', RSMTool will generate a fake score label from randomly sampled integers. This option may be used if you only need descriptive statistics for the data. 
Default: `sc1`

`test_label_column`: the label for human score used to evaluate the model. If this is set to 'fake', RSMTool will generate a fake score label from randomly sampled integers. This option may be used if you only need descriptive statistics for the data. 
Default: `sc1`

`length_column`: the label for an optional length column present in 
the train and test data. If specified, length is included 
in the inter-feature and partial correlation analyses. Note that this field
**should not** be specified if you want to use the length column as a
feature in the model. In the latter scenario, the length column will 
automatically be included in the analyses, like any other feature. If you 
specify `length_column` *and* include the same column in the feature file, 
`rsmtool` will ignore the `length_column` setting. In addition, if `length_column` has any missing values or its standard deviation is 0 (both somewhat unlikely scenarios), `rsmtool` will not include any length-based analyses in the report. 

`second_human_score_column`: the label for an optional length column present in the test data containing a second human score. If specified, additional 
information about human-human agreement and degradation will be computed and
included in the report. Note that this column must contain either numbers or be empty. Non-numeric values are not accepted. Note also that the `exclude_zero_scores` option below will apply to this column too.

`candidate_column`: the label for the column containing unique candidate IDs. Note that these are currently only used for data description. 

### Data filtering

All responses with non-numeric values in `train_label_column` or `test_label_column` and/or non-numeric values of features included into the model will be excluded from model training and evaluation. 

`exclude_zero_scores`: `true/false`. By default zero human scores will be excluded from both training and evaluation set. Set this field to `false` if you want to keep 0.
default: `true`

`flag_column`: this field makes it possible to only use responses with particular values in a given column (e.g. only responses with `0` in `ADVISORY`). The field takes a dictionary in Python format where the keys are the names of the columns and the values are lists of values for responses that will be used to train the model. E.g. `"flag_column": {"ADVISORY": 0}` will mean that the `rsmtool` will only use responses for which the `ADIVSORY` column has the numeric value `0`.  If the `flag_column` specifies several conditions (e.g. `"flag_column": {"ADVISORY": 0, "ERROR": 0}`) only responses which satisfy all conditions will be selected for further analysis  (in this example the responses where`ADVISORY` == 0 AND `ERROR` == 0).
default: `None`

`min_items_per_candidate`: integer value for the minimal number of items expected from each candidate. If any candidates have less than the specified minimal number of responses left for analysis after applying all filters, all responses from such candidates will be excluded listwise from further analysis. 
default: `None` 

### Feature set

`features`: path to the .json file with list of features and transformations. Can be absolute or relative to the location of config file. 
Default: use all columns in the file other than `spkitemid` `spkitemlab`, `itemType`, `r1`, `r2`, `score`, `sc`, `sc1`, `adj`, and the column names specified in the config file  (e.g. `length_column`, `subgroups` as well as `train_label_column` and `test_label_column`). The final set of features will be saved in the features/ folder.

`feature_subset_file`: a master file which lists all features that should be used for feature selection. The file should be in .csv format with features listed in a column named `Feature`. It can also optionally give the expected correlation between each feature and human score. This option is only meaningful in combination with `feature_subset` or `sign` below. 
Default: None

The feature list can be further constrained by using `feature_prefix` and `feature_subset`. These fields are mutually exclusive and cannot be used in the same experiment. 

`feature_subset`: The supplied feature file can specify feature subsets. These should be defined as columns in `feature_file` where the name of the column is the name of the subset and each feature is assigned 1 (included into the subset) or 0 (not included into the subset). Only one subset can be specified for each experiment. 
    
`feature_prefix`: The feature subset can also be specified by a common prefix separated by `\t`. For example, `feature_subset: 1gram, 2gram` will create a model based only on features named 1gram\t* and 2gram\t*. Several subsets can be separated by commas. 

`select_transformations`:  `true`/`false`. If this option is set to `true` the system will select the most suitable transformation based on best correlation with human score. Note that `inv` is never used for features with positive values. 
Default: `false` 

`sign`: the guidelines to scoring models require that all coefficients in the model are positive and all features have positive correlation with human score. It is possible to specify the expected correlation for each feature in `feature_subset_file`. In this case the features with expected negative correlation will be multiplied by -1 before adding them to the model. To use this option the `feature_subset_file` must contain a column named `Sign_X` where `X` is the value of `sign` field. This column can only takes `-` or `+`. 


### Score post-processing

`trim_min`: single numeric value for the lowest possible machine score. This value will be used to compute trimmed (bound) machine scores.
Default: the lowest observed human score in the training data or 1 if there are no numeric human scores.

`trim_max`: single numeric value for the highest possible machine score. This value will be used to compute trimmed (bound) machine scores.
Default: the highest observed human score in the training data or 10 if there are no numeric human scores.

`use_scaled_predictions`: `true` if you want to use scaled machine scores for the in-depth evaluations (confusion matrices, score distribution, per-prompt evaluation). Omit this field if you want to use raw scores for these evaluations. Main evaluation metrics is always computed for both scaled and raw scores. 
Default: `false`

### Subgroup analysis

`subgroups`: a list of grouping variables for generating analyses by prompt or subgroup analyses. For example, `"prompt, gender, native_language, test_country"`. These subgroup columns need to be present in both training and evaluation data. If subgroups are specified, `rsmtool` will generate: (1) description of the data by group; (2) boxplots showing feature distribution for each subgroup on the training set; and (3) tables and barplots showing system-human agreement for each subgroup on the evaluation set.
Default: no subgroups specified

### Report generation

`general_sections`: a list of general sections to be included into the final report.
See [report_sections](report_sections.md) for the list of available sections.
Default: all sections available for `rsmtool`. 

`special_sections`: a list of special sections to be included into the final report. These are the sections available to all local users via `rsmextra` package. See the documentation to `rsmextra` for the list of available sections.
Default: no special sections.

`custom_sections`: a list of custom user-defined sections (`.ipynb` files) to be included into the final report. These are the notebooks created by the user. Note that the list must contains paths to the `.ipynb` files, either absolute or relative to the config file. These notebooks have access to all of the information as described in [new_notebooks](new_notebooks.md).
Default: no custom sections.

`section_order`: a list containing the order in which the sections in the report should be generated. Note that 'section_order' must list: (a) either *all* of appropriate general sections appropriate for `rsmtool`, or a subset specified using 'sections', and (b) *all* sections specified under 'special_sections', and (c) *all* 'custom_sections' names (file name only, without the path and `.ipynb` extension).



