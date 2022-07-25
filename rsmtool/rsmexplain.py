#!/usr/bin/env python

import logging
import shap
import numpy as np
import os
import pickle
import sys
import pandas as pd
import re
from collections import OrderedDict
from skll.learner import Learner
from skll.data import Reader
from .reporter import Reporter
from .configuration_parser import configure
from .utils.logging import LogFormatter
from .utils.commandline import ConfigurationGenerator, setup_rsmcmd_parser
from .utils.constants import VALID_PARSER_SUBCOMMANDS


# utility function to get the proper feature name list we can get rid of this once the PR is done
def get_feature_names(model):
    """
    Return the features of a SKLL learner

    Parameters
    ----------
    model: skll.learner.Learner
    A SKLL Learner object.

    Returns
    -------
    feature_names: list
    The names of features used at the estimator step.

    """
    if model.feat_selector:
        return list(model.feat_vectorizer.get_feature_names_out()[model.feat_selector.get_support()])
    else:
        return list(model.feat_vectorizer.get_feature_names_out())


# utility function to get the actual array of data
def yield_ids(feature_set, range_size=None):
    """
    Grab the feature ids from from a feature set.

    Parameters
    ----------
    feature_set: skll.data.featureset.FeatureSet
    A SKLL FeatureSet.
    range_size: int or [int, int] or (int, int), optional
    A user defined range or sample size for the ids.

    Returns
    -------
    id_dic: dict
    A dictionary containing the original row-IDs for the rows sampled from the FeatureSet. The dictionary contains
    the new row indices as keys and the original FeatureSet indices as values.
    """
    id_dic = OrderedDict()
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
    Transform and vectorize features for a given learner.

    Applies the vectorizer and feature-selector step to the features in a Feature Set. Allows selection of a specific
    range of feature rows by their row-indices or random subsampling.

    Parameters
    ----------
    learner : skll.learner.Learner
    A SKLL Learner object.
    feature_set : skll.data.featureset.FeatureSet
    A SKLL FeatureSet.
    feature_range : int or [int, int] or (int, int), optional
    If feature_range is an integer, mask() will create a random subsample of feature rows. If feature_range is an
    iterable, mask() will sample a range of feature_rows using the first two integers in the iterable as indices.

    Returns
    -------
    ids : dict
    A dictionary containing the original row-IDs for the rows sampled from the FeatureSet. The dictionary contains
    the new row indices as keys and the original FeatureSet indices as values. If a random sample is created,
    this allows us to access which rows were sampled from the original set.
    features : numpy.array
    A 2D numpy array containing sampled feature rows.
    """

    ids = yield_ids(feature_set, feature_range)
    if feature_range:
        order = range(0, len(ids))
        feat_ids = [i for i in ids.values()]
        features = (learner.feat_selector.transform(
            learner.feat_vectorizer.transform(
                feature_set.vectorizer.inverse_transform(
                    feature_set.features))))
        if type(features) is not np.ndarray:
            features = features.toarray()[np.array(order), :]
        ids = dict(zip(order, feat_ids))
    else:
        features = (learner.feat_selector.transform(
            learner.feat_vectorizer.transform(
                feature_set.vectorizer.inverse_transform(
                    feature_set.features))))
        if type(features) is not np.ndarray:
            features = features.toarray()
    return ids, features


def generate_explanation(config_file_or_obj_or_dict, output_dir, logger=None):
    """
    Generate a shap.Explanation object.
    This function does all the heavy lifting. It loads the model, creates an explainer, and generates
    an explanation object. It then calls generate_report() in order to generate a SHAP report.

    Parameters
    ----------
    config_file_or_obj_or_dict : str or pathlib.Path or dict or Configuration
        Path to the experiment configuration file either a a string
        or as a ``pathlib.Path`` object. Users can also pass a
        ``Configuration`` object that is in memory or a Python dictionary
        with keys corresponding to fields in the configuration file. Given a
        configuration file, any relative paths in the configuration file
        will be interpreted relative to the location of the file. Given a
        ``Configuration`` object, relative paths will be interpreted
        relative to the ``configdir`` attribute, that _must_ be set. Given
        a dictionary, the reference path is set to the current directory.
    output_dir : str
        Path to the experiment output directory.
    logger : logging object, optional
        A logging object. If ``None`` is passed, get logger from ``__name__``.
        Defaults to ``None``.

    Raises
    ------
    FileNotFoundError
        If any of the files contained in ``config_file_or_obj_or_dict`` cannot
        be located.
    AssertionError
        If the user passes a background sample of .shape[0] < 300, rsmexplain shuts down in order to avoid inaccurate
        explanations.
    ValueError
        If config_dic["range"] cannot be converted into int or iterable of int.

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

    # we check if the background data is large enough for a meaningful representation:
    background = reader.read()
    try:
        assert background.features.shape[0] > 300
    except AssertionError:
        logger.error("Your background data set is too small. We do not recommend passing less than 300 rows of"
                     " background data. You have passed " + str(background.features.shape[0]) +
                     " rows. Shutting down...")
        sys.exit("Background sample too small, exiting program.")




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
            if row_range == 1:
                logger.error('"range" must be > 2. Shutting down.')
                exit()
            ids, data_features = mask(model, data, row_range)
        except ValueError:
            logger.info("\"range\" is not an integer, attempting to define as a range")
            try:
                row_range = re.match(r'^(\d+)[,\-\s:](\d+)$', config_dic["range"])
                if row_range:
                    logger.info('Your "range" indices have been defined as: ' + str(row_range.groups(1)))
                    index_1 = int(row_range.groups(1)[0])
                    index_2 = int(row_range.groups(1)[1])
                    if np.abs(index_1 - index_2) == 1:
                        logger.error('"range" must be > 2. Shutting down.')
                        exit()
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

    # we're generating a new explanation here, because manually munging the feature names and base values can break some
    # plots; this may be changed in newer shap versions, but under shap == 0.41. this is necessary

    explanation = shap.Explanation(explanation.values, base_values=explanation.base_values,
                                   data=explanation.data, feature_names=explanation.feature_names)

    generate_report(explanation, output_dir, ids, config_dic, logger)

    return None


def generate_report(explanation, output_dir, ids, config_dic, logger=None):
    """
    Generates a rsmexplain report.
    Generates a rsmexplain report and saves a series of files to disk. Including pickle files of the explanation object,
    and id-dictionary. Saves all shap_values to disk as .csv files.

    Parameters
    ----------
    explanation: shap.Explanation
    A shap explanation object containing shap_values, data points, feature names, base_values.
    output_dir : str
        Path to the experiment output directory
    ids: dict
    A dictionary containing the original row-IDs for the rows sampled from the FeatureSet. The dictionary contains
    the new row indices as keys and the original FeatureSet indices as values.
    config_dic : Configuration
        The Configuration object for the tool.
    logger : logging object, optional
        A logging object. If ``None`` is passed, get logger from ``__name__``.
        Defaults to ``None``.

    -------

    """
    logger = logger if logger else logging.getLogger(__name__)

    # we make sure all necessary directories exist
    os.makedirs(output_dir, exist_ok=True)
    reportdir = os.path.abspath(os.path.join(output_dir, 'report'))
    csvdir = os.path.abspath(os.path.join(output_dir, 'output'))
    figdir = os.path.abspath(os.path.join(output_dir, 'figures'))

    os.makedirs(csvdir, exist_ok=True)
    os.makedirs(figdir, exist_ok=True)
    os.makedirs(reportdir, exist_ok=True)

    # first we write the explanation object to disk, in case we need it later
    explan_path = os.path.join(csvdir, 'explanation.pkl')
    with open(explan_path, 'wb') as pickle_out:
        pickle.dump(explanation, pickle_out)
    config_dic['explanation'] = explan_path

    id_path = os.path.join(csvdir, 'ids.pkl')
    with open(id_path, 'wb') as pickle_out:
        pickle.dump(ids, pickle_out)
    config_dic['ids'] = id_path

    # now we generate a dataframe that allows us to write the shap_values to disk
    csv_path = os.path.join(csvdir, 'shap_values.csv')
    csv_path_mean = os.path.join(csvdir, 'mean_shap_values.csv')
    csv_path_max = os.path.join(csvdir, 'max_shap_values.csv')
    csv_path_min = os.path.join(csvdir, 'min_shap_values.csv')
    csv_path_abs = os.path.join(csvdir, 'abs_shap_values.csv')

    # we transform the shap values into a dataframe and save it
    shap_frame = pd.DataFrame(explanation.values, columns=explanation.feature_names, index=ids.values())
    shap_frame.to_csv(csv_path)

    # here we calculate the absolute mean shap value, turn that into a dataframe and join it with a series containing
    # the absolute max shap value
    # the feature column here serves us as an index that allows the values to be joined correctly
    # finally we also add the absolute min shap value
    means = pd.DataFrame(shap_frame.abs().mean(axis=0).sort_values(ascending=False)).join(shap_frame.abs().max(axis=0)
                                                                    .sort_values(ascending=False).rename('new_series'))
    abs_all = means.join(shap_frame.abs().min(axis=0).sort_values(ascending=False).rename('second_series'))

    # we store the mean, max, min shap values in seperate .csv files and then one joint .csv file
    shap_frame.abs().mean(axis=0).sort_values(ascending=False).to_csv(csv_path_mean, index_label='Feature', header=[
        'abs. mean shap'])
    shap_frame.abs().max(axis=0).sort_values(ascending=False).to_csv(csv_path_max, index_label='Feature', header=[
        'abs. max shap'])
    shap_frame.abs().min(axis=0).sort_values(ascending=False).to_csv(csv_path_min, index_label='Feature', header=[
        'abs. min shap'])
    abs_all.to_csv(csv_path_abs, index_label='Feature', header=['abs. mean shap', 'abs. max shap', 'abs. min shap'])
    # later we want to make some additions here to ensure that the correct indices are exported for these decisions

    # Initialize reporter
    reporter = Reporter(logger=logger)

    general_report_sections = config_dic['general_sections']

    chosen_notebook_files = reporter.get_ordered_notebook_files(general_report_sections,
                                                                context='rsmexplain')

    # add chosen notebook files to configuration
    config_dic['chosen_notebook_files'] = chosen_notebook_files

    reporter.create_explanation_report(config_dic, csvdir, reportdir)
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

        generate_explanation(os.path.abspath(args.config_file), os.path.abspath(args.output_dir))

    return None


if __name__ == '__main__':
    main()
