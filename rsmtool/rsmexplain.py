#!/usr/bin/env python

import logging
import shap
import numpy as np
import os
import pickle
import sys
import pandas as pd
import re
from skll.learner import Learner
from skll.data import Reader
from .reporter import Reporter
from .configuration_parser import configure
from .utils.logging import LogFormatter
from .utils.commandline import ConfigurationGenerator, setup_rsmcmd_parser
from .utils.constants import VALID_PARSER_SUBCOMMANDS


# utility function to get the proper feature name list we can get rid of this once the PR is done
def get_feature_names(model):
    if model.feat_selector:
        return model.feat_vectorizer.get_feature_names_out()[model.feat_selector.get_support()]
    else:
        return model.feat_vectorizer.get_feature_names_out()


# utility function to get the actual array of data
def yield_ids(feature_set, range_size=None):
    """
    Utility function that returns a dictionary object containing the indices of the data in the range
    specified.
    :param feature_set: A SKLL FeatureSet
    :param range_size: Either an integer value, or an indexable iterable containing 2 integers.
    :return: A dictionary object containing SKLL IDs as values and array indeces as keys.
    """
    id_dic = {}
    if range_size is None:
        for i in feature_set.ids:
            id_dic[np.where(feature_set.ids == i)[0][0]] = i
    elif type(range_size) is int:
        for i in shap.sample(feature_set.ids, range_size):
            id_dic[np.where(feature_set.ids == i)[0][0]] = i
    else:
        for i in feature_set.ids[range_size[0]:range_size[1]]:
            id_dic[np.where(feature_set.ids == i)[0][0]] = i
    return id_dic


def mask(learner, feature_set, feature_range=None):
    """
    Utility function that extracts features from a FeatureSet, or subsamples of those features.
    :param learner: A SKLL learner.
    :param feature_set: A SKLL FeatureSet.
    :param feature_range: Either an integer value, or an indexable iterable containing 2 integers.
    :return: A dictionary object containing SKLL IDs as values and array indeces as keys. And
    a dense numpy array of features which correspond to the dictionary row_indices in the original
    FeatureSet
    """
    ids = yield_ids(feature_set, feature_range)
    if feature_range:
        features = (learner.feat_selector.transform(
            learner.feat_vectorizer.transform(
                feature_set.vectorizer.inverse_transform(
                    feature_set.features)))).toarray()[
                   np.array([i for i in ids.keys()]), :]
    else:
        features = (learner.feat_selector.transform(
            learner.feat_vectorizer.transform(
                feature_set.vectorizer.inverse_transform(
                    feature_set.features)))).toarray()
    return ids, features


def generate_explanation(config_file_or_obj_or_dict, output_dir, logger=None):
    """
    Generates the explanation object and returns it.
    :return:
    """
    # we will implement an actual config functionality later, for now we just treat this as a dictionary
    # config_dic = config_file_or_obj_or_dic

    logger = logger if logger else logging.getLogger(__name__)

    config_dic = configure('rsmexplain', config_file_or_obj_or_dict)

    logger.info('Saving configuration file.')
    config_dic.save(output_dir)

    # first we load the model
    model = Learner.from_file(config_dic["model_path"])

    # then we load the background data
    reader = Reader.for_path(config_dic["background_data"], sparse=False, id_col=config_dic["id_column"])
    background = reader.read()

    # we load the data for predictions
    if "data" in config_dic.keys():
        data_reader = Reader.for_path(config_dic["explainable_data"], sparse=False, id_col=config_dic["id_column"])
        data = data_reader.read()
    else:
        logger.info('No "explainable_data" specified. Supplementing the background_data instead.')
        data = background

    # we grab the feature names
    feature_names = get_feature_names(model)

    # we define the background distribution
    if "background_size" in config_dic.keys():
        if config_dic["background_size"] == 'all':
            _, background_features = mask(model, background)
        elif config_dic["background_size"]:
            background_features = shap.kmeans(mask(model, background)[1], int(config_dic["background_size"]))
    else:
        logger.warning('No background sample size specified. Proceeding with a kmeans sample size 500.')
        background_features = shap.kmeans(mask(model, background)[1], 500)

    # in case the user wants a subsample explained
    if "range" in config_dic.keys() and config_dic["range"] is not None:
        try:
            row_range = int(config_dic["range"])
            ids, data_features = mask(model, data, row_range)
        except ValueError:
            logger.info("\"range\" is not an integer, attempting to define as a range")
            try:
                row_range = re.match(r'^(\d+)[,\-\s:](\d+)$', config_dic["range"])
                if row_range:
                    logger.info('Your "range" indices have been defined as: ' + str(row_range.groups(1)))
                    index_1 = int(row_range.groups(1)[0])
                    index_2 = int(row_range.groups(1)[1])
                    ids, data_features = mask(model, data, [index_1, index_2])
                else:
                    logger.error("Cannot decode the \"range\" param!")
                    raise ValueError("Cannot decode the \"range\" param!")
            except ValueError:
                logger.warning('range unspecified: This will generate explanations on the entire data set'
                               ' you pass in "explainable_data". This may take a while depending on the size of your data set.')
                ids, data_features = mask(model, data)
    else:
        logger.warning('range unspecified: This will generate explanations on the entire data set'
                       ' you pass in "explainable_data". This may take a while depending on the size of your data set.')
        ids, data_features = mask(model, data)

    # define a shap explainer
    explainer = shap.explainers.Sampling(model.model.predict, background_features, feature_names=feature_names)

    logger.info('Generating shap explanations on ' + str(data_features.shape[0]) + ' rows.')
    explanation = explainer(data_features)

    # we're doing some future-proofing here:
    if explanation.feature_names is None:
        explanation.feature_names = feature_names

    # some explainers don't correctly generate base value arrays, we take care of this here:
    try:
        explanation.base_values.shape
        if explanation.base_values.shape[0] != explanation.values.shape[0]:
            explanation.base_values = np.repeat(explanation.base_values[0], explanation.values.shape[0])
    except Exception:
        explanation.base_values = np.repeat(explanation.base_values, explanation.values.shape[0])

    generate_report(explanation, output_dir, ids)
    # Initialize reporter
    reporter = Reporter(logger=logger)

    reporter.create_explanation_report(config_dic,explanation,ids)
    return None


def generate_report(explanation, output_dir, ids):
    """
    Generates a report and saves the SHAP_values and explanation object to disk
    :return:
    """
    os.makedirs(output_dir, exist_ok=True)

    # first we write the explanation object to disk, in case we need it later
    explan_path = os.path.join(output_dir, 'explanation.pkl')
    with open(explan_path, 'wb') as pickle_out:
        pickle.dump(explanation, pickle_out)

    id_path = os.path.join(output_dir, 'ids.pkl')
    with open(id_path, 'wb') as pickle_out:
        pickle.dump(ids, pickle_out)

    # now we generate a dataframe that allows us to write the shap_values to disk
    csv_path = os.path.join(output_dir, 'shap_values.csv')
    shap_frame = pd.DataFrame(explanation.values, columns=explanation.feature_names.tolist(), index=ids.keys())
    shap_frame.to_csv(csv_path)
    # later we want to make some additions here to ensure that the correct indices are exported for these decisions

    return None


def main():
    # set up the basic logging configuration
    formatter = LogFormatter()

    # we need two handlers, one that prints to stdout
    # for the "run" command and one that prints to stderr
    # from the "generate" command; the latter is important
    # because do not want the warning to show up in the
    # generated configuration file
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)

    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)

    logging.root.setLevel(logging.INFO)
    logger = logging.getLogger(__name__)

    # setting the logger to stdout
    logging.root.addHandler(stdout_handler)

    # set up an argument parser via our helper function
    parser = setup_rsmcmd_parser('rsmexplain',
                                 uses_output_directory=True,
                                 allows_overwriting=True)
    # if we have no arguments at all then just show the help message
    if len(sys.argv) < 2:
        sys.argv.append("-h")

    # if the first argument is not one of the valid sub-commands
    # or one of the valid optional arguments, then assume that they
    # are arguments for the "run" sub-command. This allows the
    # old style command-line invocations to work without modification.
    if sys.argv[1] not in VALID_PARSER_SUBCOMMANDS + ['-h', '--help',
                                                      '-V', '--version']:
        args_to_pass = ['run'] + sys.argv[1:]
    else:
        args_to_pass = sys.argv[1:]
    args = parser.parse_args(args=args_to_pass)

    if args.subcommand == 'run':
        # when running, log to stdout
        logging.root.addHandler(stdout_handler)

        # run the experiment
        logger.info('Output directory: {}'.format(args.output_dir))

        explanation = generate_explanation(os.path.abspath(args.config_file), os.path.abspath(args.output_dir))

    return None


if __name__ == '__main__':
    main()
