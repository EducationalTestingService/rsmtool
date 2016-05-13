# Description of rsmeval config file.

This file determines the overal structure of the evaluation.
The file must be in .json format. 

## Required fields:

`experiment_id`: ID for your experiment. This can be any combination of alphanumeric values and must not contain spaces.

`predictions_file`: path to the file with predictions in .csv format. Can be absolute or relative to the location of config file. This file must contain an id column (by default `spkitemid`), a column for human scores (`sc1`) and a column for system scores. 

`system_score_column`: the label for systems scores used for evaluation.

`trim_min`: single numeric value for the lowest possible machine score. This value will be used to compute trimmed (bound) machine scores. Set this to `-100` if you do not want any additional trimming.

`trim_max`: single numeric value for the highest possible machine score. This value will be used to compute trimmed (bound) machine scores. Set this value to `100` if you do not want any additional trimming. 


## Optional fields:

`description`: A brief description of the model. This will be printed in the report. The description can contain spaces and punctuation.
Default: None

### Column names

`id_column`: the name of the column containing the response IDs. 
Default: `spkitemid`

`human_score_column`: the label for human scores used to evaluate the model.
Default: `sc1`

`second_human_score_column`: the label for an optional column present in the test data containing a second human score. If specified, additional 
information about human-human agreement and degradation will be computed and
included in the report. Note that this column must contain either numbers or be empty. Non-numeric values are not accepted. Note also that the `exclude_zero_scores` option below will apply to this column too.

`candidate_column`: the label for the column containing unique candidate IDs to compute descriptive statistics.


### Data filtering

All responses with non-numeric values in `human_score_column` or `system_score_column` will be excluded from evaluation.

`exclude_zero_scores`: `true/false`. By default zero human scores will be excluded. Set this field to `false` if you want to keep 0.
default: `true`

`flag_column`: this field makes it possible to only use responses with particular values in a given column (e.g. only responses with `0` in `ADVISORY`). The field takes a dictionary in Python format where the keys are the names of the columns and the values are lists of values for responses that will be used to train the model. E.g. `"flag_column": {"ADVISORY": 0}` will mean that the `rsmtool` will only use responses for which the `ADIVSORY` column has the numeric value `0`. If the `flag_column` specifies several conditions (e.g. `"flag_column": {"ADVISORY": 0, "ERROR": 0}`) only responses which satisfy all conditions will be selected for further analysis  (in this example the responses where`ADVISORY` == 0 AND `ERROR` == 0).
default: `None`

`min_items_per_candidate`: integer value for the minimal number of items expceted from each candidate. If any candidates have less than the specified minimal number of responses left for analysis after applying all filters, all responses from such candidates will be excluded listwise from further analysis. 
default: `None` 



### System score post-processing

`scale_with` - if the supplied scores need to be re-scaled to human range, a path to the file which contains the human scores (named `sc1`) and system scores (named `prediction`) that will be used to obtain the scaling parameters. This field can also be set to `asis` if the scores are already scaled: in this case no additional scaling will be done but the report will refer to the scores as "scaled".
Default: no additional scaling is done; the report refers to the scores as "raw".

### Subgroup analysis

`subgroups`: a list of grouping variables for generating analyses by prompt or subgroup analyses. For example, `"prompt, gender, native_language, test_country"`. These subgroup columns need to be included into the predictions file. If subgroups are specified, `rsmeval` will generate: (1) description of the data by group; and (2) tables and barplots showing system-human agreement for each subgroup on the evaluation set. However, note that if you specify a custom list in `general_sections` below, then a subgroup analysis section will *only* be included if the corresponding general section is also included. (1) corresponds to the `data_description` general section;  and (2) to the `evaluation` general section.
Default: no subgroups specified

### Report generation

`general_sections`: a list of general sections to be included into the final report. 
See [report_sections](doc/report_sections.md) for the list of available sections available for `rsmeval`.
Default: all sections available for `rsmeval`.

`special_sections`: a list of special sections to be included into the final report. These are the sections available to all local users via `rsmextra` package. See the documentation to `rsmextra` for the list of available sections.
Default: no special sections.

`custom_sections`: a list of custom user-defined sections (`.ipynb` files) to be included into the final report. These are the notebooks created by the user. Note that the list must contains paths to the `.ipynb` files, either absolute or relative to the config file. These notebooks have access to all of the information as described [new_notebooks](new_notebooks.md).
Default: no custom sections. 

`section_order`: a list containing the order in which the sections in the report should be generated. Note that 'section_order' must list: (a) either *all* of appropriate general sections appropriate for `rsmeval`, or a subset specified using 'sections', and (b) *all* sections specified under 'special_sections', and, and (c) *all* 'custom_sections' names (file name only, without the path and `.ipynb` extension).
