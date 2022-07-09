#!/usr/bin/env python

import logging
import shap
import numpy as np
from skll.learner import Learner
from skll.data import Reader
from skll.data import FeatureSet


# utility function to get the proper feature name list we can get rid of this once the PR is done
def get_feature_names(model):
    if model.feat_selector:
        return model.feat_vectorizer.get_feature_names_out()[model.feat_selector.get_support()]
    else:
        return model.feat_vectorizer.get_feature_names_out()

# utility function to get the actual array of data
def mask(learner,feature_set):
    return (learner.feat_selector.transform(
                learner.feat_vectorizer.transform(
                    feature_set.vectorizer.inverse_transform(
                        feature_set.features))))

def generate_explan(config_file_or_obj_or_dict):
    """
    Generates the explanation object and returns it.
    :return:
    """
    # we will implement an actual config functionality later, for now we just treat this as a dictionary
    config_dic = config_file_or_obj_or_dict

    # first we load the model
    model = Learner.from_file(config_dic["model_path"])

    # then we load the background data
    reader = Reader.for_path(config_dic["background_data"], sparse=False, id_col=config_dic["id_column"])
    background = reader.read()

    # we load the data for predictions
    if config_dic["data"]:
        data_reader = Reader.for_path(config_dic["data"], sparse=False, id_col=config_dic["id_column"])
        data = data_reader.read()
    else:
        data = background

    # we grab the feature names
    feature_names = get_feature_names(model)

    # we define the background distribution
    if config_dic["background_size"] == 'all':
        background_features = mask(model, background)
    elif config_dic["background_size"]:
        background_features = shap.kmeans(mask(model, background), int(config_dic["background_size"]))
    else:
        background_features = shap.kmeans(mask(model, background), 500)


    # in case the use wants a subsample explained
    if config_dic["data_size"]:
        data_features = shap.sample(mask(model, data), config_dic["data_size"])
    else:
        data_features = mask(model, data)

    # define a shap explainer
    explainer = shap.explainers.Sampling(model.model.predict, background_features, feature_names=feature_names)

    explanation = explainer(data_features)

    # we're doing some future-proofing here:
    if explanation.feature_names == None:
        explanation.feature_names = feature_names

    # some explainers don't correctly generate base value arrays, we take care of this here:
    try:
        explanation.base_values.shape
        if explanation.base_values.shape[0] != explanation.values.shape[0]:
            explanation.base_values = np.repeat(explanation.base_values[0], explanation.values.shape[0])
    except:
        explanation.base_values = np.repeat(explanation.base_values, explanation.values.shape[0])

    return explanation


def generate_report():
    """
    Generates a report and saves the SHAP_values and aexplanation object to disk
    :return:
    """
    return None


def main():
    return None


if __name__ == '__main__':
    main()
