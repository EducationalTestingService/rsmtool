# rsmcompare

`rsmcompare` generates a detailed comparison report for two models. 

Most common use cases:

* Evaluate the performance of the model after adding a new feature(s)

## Input

`rsmcompare` requires the following input:

* The output files created by `rsmtool` for both baseline ('old') and new model. The output files must be generated with the current version of the `rsmtool`. 

* Configuration file in .json format with the description of both models See [config_file_compare.md](config_file_compare.md) for further information.

## Output

`rsmcompare` produces two files: 

`*.html` - an HTML report with the comparison between two models

`*.ipnyb` - an iPython notebook used to generate the report. 

## Usage

`rsmcompare config_file [output_directory]`

* `config_file` - the path to the configuration file in .json file with the description of the data and model. 

* `output-directory` - the directory where `rsmcompare` will save its output. Default: current directory. 

