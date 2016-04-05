# Description of rsmpredict config file.

This file determines the settings used to generate predictions.
The file must be in .json format. 

## Required fields:

`experiment_id`: ID for your experiment. This should be `experiment_id` used to create the model files that you will be using for generating predictions. If you do not know the `experiment_id`, you can find it by looking at the first part of the model file names. 

`experiment_dir`: the path to the directory which contains the output of the RSMTool. This directory must contain a directory called `output` with the model files, feature pre-processing parameters and score pre-processing parameters. Can be absolute or relative to the location of config file.

`input_features_file`: the path to the file with raw feature values that will be used to generate predictions. Can be absolute or relative to the location of config file.

## Optional fields:

### Column names

`id_column`: the name of the column containing the response IDs. 
Default: `spkitemid`

### Columns that will be added to the predictions

The following columns are not used by `rsmpredict`, but can be specified so that they are included into the predictions file for subsequent use with `rsmeval`. 

`candidate_column`: the label for the column containing unique candidate IDs.

`human_score_column`: the label for human scores used to evaluate the model. This column will be renamed as `sc1`.
Default: `sc1`

`second_human_score_column`: the label for the column containing a second human score. This column will be renamed as `sc2`
Defaults: no second human score specified

`subgroups`: a list of grouping variables for generating future analyses by prompt or subgroup analyses. For example, `"prompt, gender, native_language, test_country"`.
Default: no subgroups specified

`flag_column`: see `config_rsmeval` for further information. No filtering will be done by `rsmpredict`, but the content of the flag columns will be added to the predictions file for future use with `rsmeval`.
Default: `None`






