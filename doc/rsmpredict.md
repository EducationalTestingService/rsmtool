# rsmpredict 

`rsmpredict` generates new predictions based on an existing model including feature pre-processing and score post-processing as specified by the user. Its most common use is to generate predictions for a new data set using previously trained model.

*Note*: `rsmpredict` will generate predictions for all responses in the supplied feature matrix which have numeric feature values for features included into the model. It does not require human labels. Run [rsmeval](rsmeval.md) if you have the human scores and want to evaluate the new predictions. `rsmeval` will only do the evaluations for the responses which received numeric non-zero human score. 


## Input

`rsmpredict` requires the following input:

* File with feature values. The files must be in .csv format. Each row should correspond to a single response and contain numeric feature values extracted for this response. In addition there should be a column with a unique id for each response. 

* Model files generated when training the original model. These files must be stored in the same directory. 
    
    * `_feature.csv` - preprocessing parameters

    * `_.model`, and `_.model_*.npy` - SKLL objects containing the trained model.

    *  `_postprocessing_params.csv` - post-processing parameters for predicted scores

* Configuration file in .json format with the settings for `rspredict`. See [config_file_predict.md](config_file_predict.md) for further information.

## Output

`rsmpredict` produces a .csv file with predictions for all responses in new data and two optional files:

* a .csv file with pre-processed feature values requested by using --feature flag.

The response-level predictions and the feature value files are saved in the location specified by the user. 


## Usage

`rsmpredict config_file output_file [--features feature_file]`

* config_file - the path to the configuration file in .json file with the description of the data and the model. 

* output_file - file for saving the predictions

* -- features - optional path to file for saving the pre-processed feature values for new data. 


