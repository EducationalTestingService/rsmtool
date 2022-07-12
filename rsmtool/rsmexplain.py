#!/usr/bin/env python

import logging
import shap
import numpy as np
import os
import pickle
import pandas as pd
from skll.learner import Learner
from skll.data import Reader
from .utils.logging import LogFormatter

from skll.data import FeatureSet

# this is just here for development purposes
test_config_dic = {
    "model_path": "/Users/remonitschke/Library/CloudStorage/Box-Box/ETS WORK/Stuff_for_Remo/TOEFL_Primary_writing/TP_narrative_model/toefl-primary-narrative_with-subgroups_train_all_features.csv_SVR.model",
    "background_data": "/Users/remonitschke/Library/CloudStorage/Box-Box/ETS WORK/Stuff_for_Remo/TOEFL_Primary_writing/all_features.csv",
    "id_column": "id",
    "data": "/Users/remonitschke/Library/CloudStorage/Box-Box/ETS WORK/Stuff_for_Remo/TOEFL_Primary_writing/all_features.csv",
    "data_size": 3}


# utility function to get the proper feature name list we can get rid of this once the PR is done
def get_feature_names(model):
    if model.feat_selector:
        return model.feat_vectorizer.get_feature_names_out()[model.feat_selector.get_support()]
    else:
        return model.feat_vectorizer.get_feature_names_out()


# utility function to get the actual array of data
def mask(learner, feature_set):
    return (learner.feat_selector.transform(
        learner.feat_vectorizer.transform(
            feature_set.vectorizer.inverse_transform(
                feature_set.features)))).toarray()  # most explainers require dense data representations


def generate_explanation(config_file_or_obj_or_dict, logger=None):
    """
    Generates the explanation object and returns it.
    :return:
    """
    # we will implement an actual config functionality later, for now we just treat this as a dictionary
    config_dic = config_file_or_obj_or_dict

    logger = logger if logger else logging.getLogger(__name__)

    # specifying some necessary params, allowing the logger to throw an error if they are not defined
    necessary_params = ["model_path", "background_data"]
    for i in necessary_params:
        if i not in config_dic.keys():
            logger.error('Missing Parameter Error: ', i,' is not specified in your config file')

    # first we load the model
    model = Learner.from_file(config_dic["model_path"])

    # then we load the background data
    reader = Reader.for_path(config_dic["background_data"], sparse=False, id_col=config_dic["id_column"])
    background = reader.read()

    # we load the data for predictions
    if "data" in config_dic.keys():
        data_reader = Reader.for_path(config_dic["data"], sparse=False, id_col=config_dic["id_column"])
        data = data_reader.read()
    else:
        data = background

    # we grab the feature names
    feature_names = get_feature_names(model)

    # we define the background distribution
    if "background_size" in config_dic.keys():
        if config_dic["background_size"] == 'all':
            background_features = mask(model, background)
        elif config_dic["background_size"]:
            background_features = shap.kmeans(mask(model, background), int(config_dic["background_size"]))
    else:
        logger.warning('No background sample size specified. Proceeding with a kmeans sample size 500.')
        background_features = shap.kmeans(mask(model, background), 500)

    # in case the user wants a subsample explained
    if "data_size" in config_dic.keys() and config_dic["data_size"] is not None:
        data_features = shap.sample(mask(model, data), config_dic["data_size"])
    else:
        logger.warning('data_size unspecified: This will generate explanations on the entire data set'
                       ' you pass in "data". This may take a while depending on the size of your data set.')
        data_features = mask(model, data)

    # define a shap explainer
    explainer = shap.explainers.Sampling(model.model.predict, background_features, feature_names=feature_names)

    logger.info('Generating shap explanation.')
    explanation = explainer(data_features)

    # we're doing some future-proofing here:
    if explanation.feature_names is None:
        explanation.feature_names = feature_names

    # some explainers don't correctly generate base value arrays, we take care of this here:
    try:
        explanation.base_values.shape
        if explanation.base_values.shape[0] != explanation.values.shape[0]:
            explanation.base_values = np.repeat(explanation.base_values[0], explanation.values.shape[0])
    except:
        explanation.base_values = np.repeat(explanation.base_values, explanation.values.shape[0])

    return explanation


def generate_report(explanation, output_dir):
    """
    Generates a report and saves the SHAP_values and explanation object to disk
    :return:
    """
    os.makedirs(output_dir, exist_ok=True)

    # first we write the explanation object to disk, in case we need it later
    explan_path = os.path.join(output_dir, 'explanation.pkl')
    with open(explan_path, 'wb') as pickle_out:
        pickle.dump(explanation, pickle_out)

    # now we generate a dataframe that allows us to write the shap_values to disk
    csv_path = os.path.join(output_dir, 'shap_values.csv')
    shap_frame = pd.DataFrame(explanation.values, columns=explanation.feature_names.tolist())
    shap_frame.to_csv(csv_path)
    # later we want to make some additions here to ensure that the correct indices are exported for these decisions

    return None


def main():
    explanation = generate_explanation(test_config_dic)
    generate_report(explanation, '/Users/remonitschke/rsmtool/examples/rsmexplain')
    return None


if __name__ == '__main__':
    main()
