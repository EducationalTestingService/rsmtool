"""
Functions dealing with finding and reading input files.

:author: Nitin Madnani (nmadnani@ets.org)
:author: Anastassia Loukina (aloukina@ets.org)
:organization: ETS
"""

import json
import logging
import re
import warnings

import pandas as pd

from collections import defaultdict
from os.path import abspath, basename, dirname, exists, join

from numpy import nan
from numpy.random import RandomState

from rsmtool import HAS_RSMEXTRA
from rsmtool.create_features import (generate_feature_names,
                                     generate_default_specs,
                                     generate_specs_from_data)
from rsmtool.model import check_model_name
from rsmtool.preprocess import (filter_on_column,
                                filter_on_flag_columns)
from rsmtool.report import get_ordered_notebook_files

if HAS_RSMEXTRA:
    from rsmextra.settings import (default_feature_subset_file,
                                   default_feature_sign)

def locate_custom_sections(custom_report_section_paths, configpath):
    """
    Get the absolute paths for custom report sections
    and check that the files exist
    """
    custom_report_sections = []
    for cs_path in custom_report_section_paths:
        cs_location = locate_file(cs_path, configpath)
        if not cs_location:
            raise FileNotFoundError("Error: custom section not found at "
                                    "{}.".format(cs_path))
        else:
            custom_report_sections.append(cs_location)
    return custom_report_sections


def normalize_json_fields(json_obj):
    """
    Normalize the the field names in `json_obj` in order to
    maintain backwards compatibility with old config files
    """
    logger = logging.getLogger(__name__)

    field_name_mapping = {'expID': 'experiment_id',
                          'LRmodel': 'model',
                          'train': 'train_file',
                          'test': 'test_file',
                          'predictions': 'predictions_file',
                          'feature': 'features',
                          'train.lab': 'train_label_column',
                          'test.lab': 'test_label_column',
                          'trim.min': 'trim_min',
                          'trim.max': 'trim_max',
                          'scale': 'use_scaled_predictions',
                          'feature.subset': 'feature_subset'}

    model_name_mapping = {'empWt':'LinearRegression',
                          'eqWt': 'EqualWeightsLR',
                          'empWtBalanced': 'RebalancedLR',
                          'empWtDropNeg': '',
                          'empWtNNLS': 'NNLR',
                          'empWtDropNegLasso': 'LassoFixedLambdaThenNNLR',
                          'empWtLasso': 'LassoFixedLambdaThenLR',
                          'empWtLassoBest': 'PositiveLassoCVThenLR',
                          'lassoWtLasso': 'LassoFixedLambda',
                          'lassoWtLassoBest': 'PositiveLassoCV'}

    # Create a new json object with the normalized field names
    new_json_obj = {}

    for field_name in json_obj:
        if field_name in field_name_mapping:
            norm_field_name = field_name_mapping[field_name]
            warnings.warn("""The field name "{}" is deprecated and will be """
                          """removed  in a future release, please use the """
                          """new field name "{}" instead.""".format(field_name,
                                                                    norm_field_name),
                          category=DeprecationWarning)
        else:
            norm_field_name = field_name
        new_json_obj[norm_field_name] = json_obj[field_name]

    # Convert old values for prediction scaling:
    if 'use_scaled_predictions' in new_json_obj:
        if new_json_obj['use_scaled_predictions'] in ['scale', True]:
                new_json_obj['use_scaled_predictions'] = True
        elif new_json_obj['use_scaled_predictions'] in ['raw', False]:
                new_json_obj['use_scaled_predictions'] = False
        else:
            raise ValueError("Please use the new format "
                             "to specify prediction scaling:\n "
                             "'use_scaled_predictions': true/false")

    # convert old model names to new ones, if we have them
    if 'model' in new_json_obj:
        model_name = new_json_obj['model']

        if model_name == 'empWtDropNeg':
        # if someone is using `empWtDropNeg`, we tell them that it is
        # no longer available and they should be using NNLR instead.
            logger.error("""The model name "empWtDropNeg" is no """
                         """longer available, please use the equivalent """
                         """model "NNLR" instead.""")

        # otherwise, just raise a deprecation warning if they are using
        # an old model name
        elif model_name in model_name_mapping:
            norm_model_name = model_name_mapping[model_name]
            warnings.warn("""The model name "{}" is deprecated and will be """
                          """removed  in a future release, please use the """
                          """new model name "{}" instead.""".format(model_name,
                                                                    norm_model_name),
                          category=DeprecationWarning)
            new_json_obj['model'] = norm_model_name

    return new_json_obj


def validate_and_populate_json_fields(json_obj, context='rsmtool'):
    """
    Ensure that all required fields are specified,
    add default to values for all other fields, and
    ensure that all specified fields are valid.
    """

    logger = logging.getLogger(__name__)

    new_json_obj = json_obj.copy()

    # 1. Check to make sure all required fields are specified
    if context == 'rsmtool':
        required_fields = ['experiment_id',
                           'model', 'train_file',
                           'test_file']
    elif context == 'rsmeval':
        required_fields = ['experiment_id',
                           'predictions_file',
                           'system_score_column',
                           'trim_min', 'trim_max']
    elif context == 'rsmpredict':
        required_fields = ['experiment_id',
                           'experiment_dir',
                           'input_features_file']
    elif context == 'rsmcompare':
        required_fields = ['experiment_id_old', 'experiment_dir_old',
                           'experiment_id_new', 'experiment_dir_new']

    for field in required_fields:
        if field not in new_json_obj:
            raise ValueError("The config file must specify '{}'".format(field))

    # 2. Add default values for all unspecified fields
    defaults = {'id_column': 'spkitemid',
                'description': '',
                'description_old': '',
                'description_new': '',
                'train_label_column': 'sc1',
                'test_label_column': 'sc1',
                'human_score_column': 'sc1',
                'exclude_zero_scores': True,
                'use_scaled_predictions': False,
                'use_scaled_predictions_old': False,
                'use_scaled_predictions_new': False,
                'select_transformations': False,
                'scale_with': None,
                'sign': None,
                'features': None,
                'length_column': None,
                'second_human_score_column': None,
                'form_level_scores': None,
                'candidate_column': None,
                'general_sections': 'all',
                'special_sections': None,
                'custom_sections': None,
                'feature_subset_file': None,
                'feature_subset': None,
                'feature_prefix': None,
                'trim_min': None,
                'trim_max': None,
                'subgroups': [],
                'section_order': None,
                'flag_column': None}

    for field in defaults:
        if field not in new_json_obj:
            new_json_obj[field] = defaults[field]

    # 3. Check to make sure no unrecognized fields are specified
    for field in new_json_obj:
        if field not in defaults and field not in required_fields:
            raise ValueError("Unrecognized field '{}' in json file".format(field))

    # 4. Check for fields that must be specified together and try to use the default feature file:
    if new_json_obj['feature_subset'] and not new_json_obj['feature_subset_file']:

        # Check if we have the default subset file from rsmextra
        if HAS_RSMEXTRA:
            new_json_obj['feature_subset_file'] = default_feature_subset_file
            logger.warning("You requested feature subsets but did not "
                           "specify any feature file. "
                           "The tool will use the default feature file {} "
                           "available via rsmextra".format(basename(default_feature_subset_file)))
        else:
            raise ValueError("If you want to use feature subsets, you "
                             "must specify a feature subset file")

    if new_json_obj['sign'] and not new_json_obj['feature_subset_file']:

        # Check if we have the default subset file from rsmextra
        if HAS_RSMEXTRA:
            new_json_obj['feature_subset_file'] = default_feature_subset_file
            logger.warning("You specified the expected sign of correlation but did not "
                           "specify a feature subset file. The tool will use "
                           "the default feature subset file {} "
                           "available via rsmextra".format(basename(default_feature_subset_file)))
        else:
            raise ValueError("If you want to specify the expected sign of correlation "
                             "for each feature, you must specify a feature subset file")

    # Use the default sign if we are using the default feature file
    # and sign has not been specified in the config file
    if HAS_RSMEXTRA:
        if (new_json_obj['feature_subset_file'] == default_feature_subset_file
            and not new_json_obj['sign']):
            new_json_obj['sign'] = default_feature_sign

    # 5. Check the fields that requires rsmextra
    if not HAS_RSMEXTRA:
        if new_json_obj['special_sections']:
            raise ValueError("Special sections are only available to ETS"
                             " users by installing the rsmextra package.")

    return new_json_obj


def process_json_fields(json_obj):
    """
    Converts json fields which are read in
    string to an appropriate format. Fields which
    can take multiple values lists are converted to lists
    if they have not been already formatted as such.
    """
    # convert multiple values into lists
    list_fields = ['feature_prefix',
                   'general_sections',
                   'special_sections',
                   'custom_sections',
                   'subgroups', 'section_order']

    new_json_obj = json_obj.copy()

    for field in list_fields:
        if new_json_obj[field]:
            if type(new_json_obj[field]) != list:
                new_json_obj[field] = new_json_obj[field].split(',')
                new_json_obj[field] = [prefix.strip() for prefix in new_json_obj[field]]

    return new_json_obj

# method to parse json config file but removes the comments first
# (http://www.lifl.fr/~riquetd/parse-a-json-file-with-comments.html)
def parse_json_with_comments(filename):
    """ Parse a JSON file
        First remove comments and then use the json module package
        Comments look like :
            // ...
        or
            /*
            ...
            */
    """
    # Regular expression to identify comments
    comment_re = re.compile(
        '(^)?[^\S\n]*/(?:\*(.*?)\*/[^\S\n]*|/[^\n]*)($)?',
        re.DOTALL | re.MULTILINE
    )

    with open(filename) as f:
        content = ''.join(f.readlines())

        ## Looking for comments
        match = comment_re.search(content)
        while match:
            # single line comment
            content = content[:match.start()] + content[match.end():]
            match = comment_re.search(content)

        # Return json file
        return json.loads(content)


def normalize_and_validate_feature_file(feature_json):

    """
    Normalize the the field names in `feature_json`
    in order to maintain backwards compatibility
    with old config files.
    """
    field_mapping = {'wt': 'sign',
                     'featN': 'feature',
                     'trans': 'transform'}

    required_fields = ['feature', 'sign', 'transform']

    new_json_obj = defaultdict(list)

    feature_list = feature_json['features'] if 'features' in feature_json else feature_json['feats']

    for feature_dict in feature_list:
        new_feature_dict = {}
        for field in feature_dict:
            norm_field = field_mapping[field] if field in field_mapping else field
            new_feature_dict[norm_field] = feature_dict[field]

        missing_fields = set(required_fields).difference(new_feature_dict.keys())
        if missing_fields:
            raise KeyError("The feature file does not "
                           "contain the following fields: {}".format(','.join(missing_fields)))

        # make sure that the sign is float
        new_feature_dict['sign'] = float(new_feature_dict['sign'])
        new_json_obj['features'].append(new_feature_dict)

    # check to make sure that there are no duplicate feature names
    feature_names = [feature_dict['feature'] for feature_dict in new_json_obj['features']]
    duplicate_features = [name for name in set(feature_names)
                          if feature_names.count(name) > 1]
    if duplicate_features:
        raise ValueError("The following feature names are duplicated "
                         "in the feature file: {}".format(duplicate_features))

    return new_json_obj

def read_json_file(json_file):
    try:
        obj = parse_json_with_comments(json_file)
    except ValueError:
        raise ValueError("The main configuration file '{}' exists but "
                         "is formatted incorrectly. Please check that "
                         "each line ends with a comma, there is no comma "
                         "at the end of the last line, and that all quotes "
                         "match.".format(json_file))
    return obj


def check_main_config(obj, context='rsmtool'):

    """
    Normalize all fields in main config file, check for
    all required fields, add the default values, and convert
    all strings to appropriate objects.
    """
    obj = normalize_json_fields(obj)
    obj = validate_and_populate_json_fields(obj, context=context)
    obj = process_json_fields(obj)
    return obj


def locate_file(filepath, configpath):

    """
    Try to locate an experiment file. If the given path
    doesn't exist, then may be the path is relative to
    the path of the config file. If neither exists, then
    return None.
    """

    # the feature config file can be in the 'feature' directory
    # at the same level as the main config file
    alternate_path = abspath(join(configpath, filepath))

    retval = None

    # if the given path exists as is, convert
    # that to an absolute path and return
    if exists(filepath):
        retval = abspath(filepath)

    # otherwise check if it exists relative
    # to the directory that contains the main config file
    elif exists(alternate_path):
        retval = alternate_path

    return retval


def check_subgroups(df, subgroups):

    """
    Check that all subgroups, if specified, correspond to
    columns in the provided data frame, and replace all NaNs
    in subgroups values with 'No info' for later convenience.
    """

    missing_subgroup_columns = set(subgroups).difference(df.columns)
    if missing_subgroup_columns:
        raise KeyError("The data does not contain columns "
                       "for all subgroups specified in the "
                       "configuration file. Please check for "
                       "capitalization and other spelling "
                       "errors and make sure the subgroup "
                       "names do not contain hyphens. "
                       "The data does not have columns "
                       "for the following "
                       "subgroups: {}".format(', '.join(missing_subgroup_columns)))

    # replace any empty values in subgroups values by "No info"
    empty_value = re.compile(r"^\s*$")
    df[subgroups] = df[subgroups].replace(to_replace=empty_value, value='No info')
    return(df)



def get_trim_min_max(config_obj):

    """
    Get the specified trim min and max, if any and make sure
    they are numeric.
    """

    spec_trim_min = config_obj.get('trim_min', None)
    spec_trim_max = config_obj.get('trim_max', None)

    if spec_trim_min:
        spec_trim_min = float(spec_trim_min)
    if spec_trim_max:
        spec_trim_max = float(spec_trim_max)
    return(spec_trim_min, spec_trim_max)


def check_flag_column(config_obj):

    """
    make sure the flag_column field is in the correct format.
    Get flag columns and values for filtering if any
    and convert single values to lists
    """

    logger = logging.getLogger(__name__)

    new_filtering_dict = {}
    if config_obj['flag_column']:

        original_filtering_dict = config_obj['flag_column']

        # first check that the value is a dictionary
        if type(original_filtering_dict) != dict:
            raise ValueError("'flag_column' must be a dictionary. "
                             "Please refer to the documentation for further "
                             "information")

        for column in original_filtering_dict:

            # if we were given a single value, convert it to list
            if type(original_filtering_dict[column]) != list:
                new_filtering_dict[column] = [original_filtering_dict[column]]
                logger.warning("The filtering condition {} for column {} was "
                               "converted to list. Only responses where "
                               "{} == {} will be used for training and/or "
                               "evaluating the model. You can ignore this "
                               "warning if this is the correct interpretation "
                               "of your configuration settings".format(original_filtering_dict[column],
                                                                       column,
                                                                       column,
                                                                       original_filtering_dict[column]))
            else:
                new_filtering_dict[column] = original_filtering_dict[column]
                logger.info("Only responses where "
                            "{} equals one of the following values "
                            "will be used for training and/or "
                            "evaluating the model: {}.".format(column,
                                                               ', '.join(map(str, original_filtering_dict[column]))))

    return new_filtering_dict

def check_feature_subset_file(feature_specs,
                              subset=None,
                              sign=None):

    # check that the file is in the correct format and contains all the requested values
    if 'Feature' not in feature_specs.columns:
        raise ValueError("The feature_subset_file must contain a column named 'Feature' "
                         "containing the feature names.")
    if subset:
        if subset not in feature_specs.columns:
            raise ValueError("Unknown value for feature_subset: {}".format(subset))

        if not feature_specs[subset].isin([0, 1]).all():
            raise ValueError("The subset columns in feature file can only contain 0 or 1")

    if sign:
        if not 'Sign_'+sign in feature_specs.columns:
                raise ValueError("The feature_subset_file does not contain the requested "
                                 "sign column {}".format('Sign_'+sign))

        if not feature_specs[subset].isin(['-', '+']).all():
            raise ValueError("The sign columns in feature file can only contain - or +")



def load_experiment_data(main_config_file, outdir):

    """
    Set up the experiment by loading the training
    and evaluation data sets and preprocessing them.
    """

    logger = logging.getLogger(__name__)

    # read in the main config file
    logger.info('Reading configuration file: {}'.format(main_config_file))
    config_obj = read_json_file(main_config_file)
    config_obj = check_main_config(config_obj)

    # get the directory where the config file lives
    configpath = dirname(main_config_file)

    # get the experiment ID
    experiment_id = config_obj['experiment_id']

    # get the description
    description = config_obj['description']

    # get the column name for the labels for the training and testing data
    train_label_column = config_obj['train_label_column']
    test_label_column = config_obj['test_label_column']

    # get the column name that will hold the ID for
    # both the training and the test data
    id_column = config_obj['id_column']

    # get the specified trim min and max values
    spec_trim_min, spec_trim_max = get_trim_min_max(config_obj)

    # get the name of the optional column that
    # contains response length.
    length_column = config_obj['length_column']

    # get the name of the optional column that
    # contains the second human score
    second_human_score_column = config_obj['second_human_score_column']

    # get the name of the optional column that
    # contains the candidate ID
    candidate_column = config_obj['candidate_column']

    # if the test label column is the same as the
    # second human score column, raise an error
    if test_label_column == second_human_score_column:
        raise ValueError("'test_label_column' and "
                         "'second_human_score_column' cannot have the "
                         "same value.")

    # get the name of the model that we want to train and
    # check that it's valid
    model_name = config_obj['model']
    model_type = check_model_name(model_name)

    # are we excluding zero scores?
    exclude_zero_scores = config_obj['exclude_zero_scores']

    # if we are excluding zero scores but trim_min
    # is set to 0, then we need to warn the user
    if exclude_zero_scores and spec_trim_min == 0:
        logger.warning("'exclude_zero_scores' is set to True but "
                       "'trim_min' is set to 0. This may cause "
                       " unexpected behavior.")

    # are we filtering on any other columns?
    flag_column_dict = check_flag_column(config_obj)

    # are we generating fake labels?
    use_fake_train_labels = train_label_column == 'fake'
    use_fake_test_labels = test_label_column == 'fake'

    # are we analyzing scaled or raw prediction values
    use_scaled_predictions = config_obj['use_scaled_predictions']

    # get the subgroups if any
    subgroups = config_obj.get('subgroups')

    # are there specific general report sections we want to include?
    general_report_sections = config_obj['general_sections']

    # what about the special or custom sections?
    special_report_sections = config_obj['special_sections']

    custom_report_section_paths = config_obj['custom_sections']

    if custom_report_section_paths:
        logger.info('Locating custom report sections')
        custom_report_sections = locate_custom_sections(custom_report_section_paths,
                                                         configpath)
    else:
        custom_report_sections = []

    section_order = config_obj['section_order']

    chosen_notebook_files = get_ordered_notebook_files(general_report_sections,
                                                       special_report_sections,
                                                       custom_report_sections,
                                                       section_order,
                                                       subgroups,
                                                       model_type=model_type,
                                                       context='rsmtool')

    # Read in the feature configurations.
    # Location of feature file
    feature_field = config_obj['features']

    # Check whether feature subset file exists and whether we are using
    # feature subset of prefix
    feature_subset_file = config_obj['feature_subset_file']
    if feature_subset_file:
        feature_subset_file_location = locate_file(feature_subset_file, configpath)
        if not feature_subset_file_location:
            raise FileNotFoundError('Feature subset file {} not '
                                    'found.\n'.format(config_obj['feature_subset_file']))

    feature_subset = config_obj['feature_subset']
    feature_prefix = config_obj['feature_prefix']

    # if the user requested feature_subset file and feature subset,
    # read the file and check its format
    if feature_subset_file and feature_subset:
        feature_subset_specs = pd.read_csv(feature_subset_file_location)
        check_feature_subset_file(feature_subset_specs, feature_subset)
    else:
        feature_subset_specs = None


    # Do we need to automatically find the best transformations/change sign?
    select_transformations = config_obj['select_transformations']
    feature_sign = config_obj['sign']

    requested_features = []
    feature_specs = {}
    select_features_automatically = True

    # For backward compatibility, we check whether this field can
    # be set to all and set the select_transformations to true
    # as was done in the previous version.
    if feature_field == 'all':
        select_transformations = True
    elif feature_field is not None:
        feature_file_location = locate_file(feature_field, configpath)
        select_features_automatically = False
        if not feature_file_location:
            raise FileNotFoundError('Feature file {} not '
                                    'found.\n'.format(config_obj['features']))
        else:
            logger.info('Reading feature file: {}'.format(feature_file_location))
            feature_json = read_json_file(feature_file_location)
            feature_specs = normalize_and_validate_feature_file(feature_json)
            requested_features = [fdict['feature'] for fdict in feature_specs['features']]

    # check to make sure that `length_column` or `second_human_score_column`
    # are not also included in the requested features, if they are specified
    if (length_column and
        length_column in requested_features):
        raise ValueError("The value of 'length_column' ('{}') cannot be "
                         "used as a model feature.".format(length_column))

    if (second_human_score_column and
        second_human_score_column in requested_features):
        raise ValueError("The value of 'second_human_score_column' ('{}') cannot be "
                         "used as a model feature.".format(second_human_score_column))

    # Specify column names that cannot be used as features
    reserved_column_names = list(set(['spkitemid', 'spkitemlab',
                                      'itemType', 'r1', 'r2', 'score',
                                      'sc', 'sc1', 'adj',
                                      train_label_column,
                                      test_label_column,
                                      id_column] + subgroups + list(flag_column_dict.keys())))

    # if `second_human_score_column` is specified, then
    # we need to add `sc2` to the list of reserved column
    # names. And same for 'length' and 'candidate', if `length_column`
    # and `candidate_column` are specified
    if second_human_score_column:
        reserved_column_names.append('sc2')
    if length_column:
        reserved_column_names.append('length')
    if candidate_column:
        reserved_column_names.append('candidate')

    # Make sure that the training data as specified in the
    # config file actually exists on disk and if it does,
    # load it and filter out the bad rows and features with
    # zero standard deviation. Also double check that the requested
    # features exist in the data or obtain the feature names if
    # no feature file was given.
    train_file_location = locate_file(config_obj['train_file'], configpath)
    if not train_file_location:
        raise FileNotFoundError('Error: Training file {} '
                                'not found.\n'.format(config_obj['train_file']))
    else:
        logger.info('Reading training data: {}'.format(train_file_location))

    (df_train_features,
     df_train_metadata,
     df_train_other_columns,
     df_train_excluded,
     df_train_length,
     _,
     df_train_flagged_responses,
     used_trim_min,
     used_trim_max,
     feature_names) = load_and_filter_data(train_file_location,
                                           train_label_column,
                                           id_column,
                                           length_column,
                                           None,
                                           candidate_column,
                                           requested_features,
                                           reserved_column_names,
                                           spec_trim_min,
                                           spec_trim_max,
                                           flag_column_dict,
                                           subgroups,
                                           exclude_zero_scores=exclude_zero_scores,
                                           exclude_zero_sd=True,
                                           feature_subset_specs=feature_subset_specs,
                                           feature_subset=feature_subset,
                                           feature_prefix=feature_prefix,
                                           use_fake_labels=use_fake_train_labels)

    # Generate feature specifications now that we
    # know what features are selected
    if select_features_automatically:
        if select_transformations is False:
            feature_specs = generate_default_specs(feature_names)
        else:
            feature_specs = generate_specs_from_data(feature_names,
                                                     'sc1',
                                                     df_train_features,
                                                     feature_subset_specs=feature_subset_specs,
                                                     feature_sign=feature_sign)
    # Sanity check to make sure the function returned the
    # same feature names as specified in feature json file,
    # if there was one
    elif not select_features_automatically:
        assert feature_names == requested_features

    # Do the same for the test data except we can ignore the trim min
    # and max since we already have that from the training data and
    # we have the feature_names when no feature file was specified.
    # We also allow features with 0 standard deviation in the test file.
    test_file_location = locate_file(config_obj['test_file'], configpath)
    if not test_file_location:
        raise FileNotFoundError('Error: Evaluation file '
                                '{} not found.\n'.format(config_obj['test_file']))
    elif (test_file_location == train_file_location
            and train_label_column == test_label_column):
        logging.warning('The same data file and label '
                        'column are used for both training '
                        'and evaluating the model. No second '
                        'score analysis will be performed, even '
                        'if requested.')
        df_test_features = df_train_features.copy()
        df_test_metadata = df_train_metadata.copy()
        df_test_excluded = df_train_excluded.copy()
        df_test_other_columns = df_train_other_columns.copy()
        df_test_flagged_responses = df_train_flagged_responses.copy()
        df_test_human_scores = pd.DataFrame()
    else:
        logger.info('Reading evaluation data: {}'.format(test_file_location))
        (df_test_features,
         df_test_metadata,
         df_test_other_columns,
         df_test_excluded,
         _,
         df_test_human_scores,
         df_test_flagged_responses,
         _, _, _) = load_and_filter_data(test_file_location,
                                         test_label_column,
                                         id_column,
                                         None,
                                         second_human_score_column,
                                         candidate_column,
                                         feature_names,
                                         reserved_column_names,
                                         used_trim_min,
                                         used_trim_max,
                                         flag_column_dict,
                                         subgroups,
                                         exclude_zero_scores=exclude_zero_scores,
                                         exclude_zero_sd=False,
                                         use_fake_labels=use_fake_test_labels)

    return (df_train_features, df_test_features,
            df_train_metadata, df_test_metadata,
            df_train_other_columns, df_test_other_columns,
            df_train_excluded, df_test_excluded,
            df_train_length, df_test_human_scores,
            df_train_flagged_responses,
            df_test_flagged_responses,
            experiment_id, description,
            train_file_location, test_file_location,
            feature_specs, model_name, model_type,
            train_label_column, test_label_column,
            id_column, length_column, second_human_score_column,
            candidate_column,
            subgroups,
            feature_subset_file,
            used_trim_min, used_trim_max,
            use_scaled_predictions, exclude_zero_scores,
            select_features_automatically,
            chosen_notebook_files)


def load_and_filter_data(csv_file,
                         label_column,
                         id_column,
                         length_column,
                         second_human_score_column,
                         candidate_column,
                         requested_feature_names,
                         reserved_column_names,
                         given_trim_min,
                         given_trim_max,
                         flag_column_dict,
                         subgroups,
                         exclude_zero_scores=True,
                         exclude_zero_sd=False,
                         feature_subset_specs=None,
                         feature_subset=None,
                         feature_prefix=None,
                         use_fake_labels=False):
    """
    Load the data from `csv_file` and filters it to remove
    rows that have zero/non-numeric values for `label_column`.
    If feature_names are specified, it checks whether any
    features that are specifically requested in `feature_names`
    are missing from the data. If no feature_names are specified,
    these are generated based on column names and subset information
    if available. The function then excludes non-numeric values for
    any feature. It also generates fake labels between 1 and 10 if
    `use_fake_parameters` is set to True. Finally, it renames the id
    and label column and splits the data into the data frame with
    feature values and score label and the data frame with other
    available metadata.
    """

    logger = logging.getLogger(__name__)

    # read the csv file into a data frame but we want to make
    # sure to read in the `id_column`, `candidate_column` and
    # subgroups (if any) as a string to ensure
    # that we do not lose information, e.g., initial zeros
    string_columns = [id_column, candidate_column] + subgroups
    converter_dict = dict([(column, str) for column in string_columns if column])

    # read in the CSV file
    df = pd.read_csv(csv_file, converters=converter_dict)

    # make sure that the columns specified in the config file actually exist
    columns_to_check = [id_column, label_column]

    if length_column:
        columns_to_check.append(length_column)

    if second_human_score_column:
        columns_to_check.append(second_human_score_column)

    missing_columns = set(columns_to_check).difference(df.columns)
    if missing_columns:
        raise KeyError("Columns {} from the config file "
                       "do not exist in the data.".format(missing_columns))

    # it is possible for the `id_column` and `candidate_column` to be
    # set to the same column name in the CSV file, e.g., if there is
    # only one response per candidate. If this happens, we neeed to
    # create a duplicate column for candidates or id for the downstream
    # processing to work as usual.
    if id_column == candidate_column:
        # if the name for both columns is `candidate`, we need to
        # create a separate id_column name
        if id_column == 'candidate':
            df['spkitemid'] = df['candidate'].copy()
            id_column = 'spkitemid'
        # else we create a separate `candidate` column
        else:
            df['candidate'] = df[id_column].copy()
            candidate_column = 'candidate'

    df = rename_default_columns(df,
                                requested_feature_names,
                                id_column,
                                label_column,
                                second_human_score_column,
                                length_column,
                                None,
                                candidate_column)

    # check that the id_column contains unique values
    if df['spkitemid'].size != df['spkitemid'].unique().size:
        raise ValueError("The data contains duplicate response IDs in "
                         "'{}'. Please make sure all response IDs are "
                         "unique and re-run the tool.".format(id_column))

    # Generate feature names if no feature .json file was provided.
    if len(requested_feature_names) == 0:
        feature_names = generate_feature_names(df,
                                               reserved_column_names,
                                               feature_subset_specs=feature_subset_specs,
                                               feature_subset=feature_subset,
                                               feature_prefix=feature_prefix)
    else:
        feature_names = requested_feature_names

    # make sure that feature names do not contain reserved column names
    illegal_feature_names = set(feature_names).intersection(reserved_column_names)
    if illegal_feature_names:
                raise ValueError("The following reserved column names "
                                 "cannot be used as feature names: '{}'. "
                                 "Please rename these columns and "
                                 "re-run the experiment.".format(', '.join(illegal_feature_names)))

    # check to make sure that the subgroup columns are all present
    df = check_subgroups(df, subgroups)

    # filter out the responses based on flag columns
    (df_responses_with_requested_flags,
     df_responses_with_excluded_flags) = filter_on_flag_columns(df, flag_column_dict)

    # filter out the rows that have non-numeric or zero labels
    # unless we are going to generate fake labels in the first place
    if not use_fake_labels:
        (df_filtered,
         df_excluded) = filter_on_column(df_responses_with_requested_flags,
                                         'sc1',
                                         'spkitemid',
                                         exclude_zeros=exclude_zero_scores)

        # make sure that the remaining data frame is not empty
        if len(df_filtered) == 0:
            raise ValueError("No responses remaining after filtering out "
                             "non-numeric human scores. No further analysis "
                             "can be run. ")

        trim_min = given_trim_min if given_trim_min else df_filtered['sc1'].min()
        trim_max = given_trim_max if given_trim_max else df_filtered['sc1'].max()
    else:
        df_filtered = df_responses_with_requested_flags.copy()
        trim_min = given_trim_min if given_trim_min else 1
        trim_max = given_trim_max if given_trim_max else 10
        logger.info("Generating labels randomly "
                    "from [{}, {}]".format(trim_min, trim_max))
        randgen = RandomState(seed=1234567890)
        df_filtered[label_column] = randgen.random_integers(trim_min,
                                                            trim_max,
                                                            size=len(df_filtered))

    # make sure there are no missing features in the data
    missing_features = set(feature_names).difference(df_filtered.columns)
    if not missing_features:
        # make sure all features selected for model building are numeric
        # and also replace any non-numeric feature values in already
        # excluded data with NaNs for consistency
        for feat in feature_names:
            df_excluded[feat] = pd.to_numeric(df_excluded[feat], errors='coerce').astype(float)
            newdf, newdf_excluded = filter_on_column(df_filtered,
                                                     feat,
                                                     'spkitemid',
                                                     exclude_zeros=False,
                                                     exclude_zero_sd=exclude_zero_sd)
            del df_filtered
            df_filtered = newdf
            df_excluded = pd.merge(df_excluded, newdf_excluded, how='outer')

        # make sure that the remaining data frame is not empty
        if len(df_filtered) == 0:
            raise ValueError("No responses remaining after filtering "
                             "out non-numeric feature values. No further "
                             "analysis can be run.")

        # Raise warning if we excluded features that were
        # specified in the .json file because sd == 0.
        omitted_features = set(requested_feature_names).difference(df_filtered.columns)
        if omitted_features:
            logger.warning("The following requested features "
                           "were excluded because their standard "
                           "deviation on the training set was 0: {}.\n"
                           "Please edit the feature file to exclude "
                           "these features and re-run the "
                           "tool".format(', '.join(omitted_features)))
        # Update the feature names
        feature_names = [feature for feature in feature_names
                         if feature in df_filtered]
    else:
        raise KeyError("{} does not contain "
                       "columns for all features specified in "
                       "the feature file. Please check for "
                       "capitalization and other spelling "
                       "errors and make sure the feature "
                       "names do not contain hyphens. "
                       "The data does not have columns "
                       "for the following features: "
                       "{}".format(csv_file,
                                   ', '.join(missing_features)))

    # check the values for length column. We do this after filtering
    # to make sure we have removed responses that have not been
    # processed correctly. Else rename length column to
    # ##ORIGINAL_NAME##.
    if (length_column and
        (len(df_filtered[df_filtered['length'].isnull()]) != 0 or
             df_filtered['length'].std() <= 0)):
        logger.warning("The {} column either has missing values or a standard"
                       " deviation <= 0. No length-based analysis will be"
                       " provided. The column will be renamed as ##{}## and"
                       " saved in *train_other_columns.csv.".format(length_column,
                                                                    length_column))
        df_filtered.rename(columns={'length': '##{}##'.format(length_column)},
                           inplace=True)

    # create separate data-frames for features and sc1, all other
    # information, and responses excluded during filtering
    not_other_columns = set()
    feature_columns = ['spkitemid', 'sc1'] + feature_names
    df_filtered_features = df_filtered[feature_columns]
    not_other_columns.update(feature_columns)

    metadata_columns = ['spkitemid'] + subgroups
    if candidate_column:
        metadata_columns.append('candidate')
    df_filtered_metadata = df_filtered[metadata_columns]
    not_other_columns.update(metadata_columns)

    df_filtered_length = pd.DataFrame()
    length_columns = ['spkitemid', 'length']
    if length_column and 'length' in df_filtered:
        df_filtered_length = df_filtered[length_columns]
        not_other_columns.update(length_columns)

    df_filtered_human_scores = pd.DataFrame()
    human_score_columns = ['spkitemid', 'sc1', 'sc2']
    if second_human_score_column and 'sc2' in df_filtered:
        df_filtered_human_scores = df_filtered[human_score_columns].copy()
        not_other_columns.update(['sc2'])
        # filter out any non-numeric value rows
        # as well as zeros, if we were asked to
        df_filtered_human_scores['sc2'] = pd.to_numeric(df_filtered_human_scores['sc2'],
                                                        errors='coerce').astype(float)
        if exclude_zero_scores:
            df_filtered_human_scores['sc2'] = df_filtered_human_scores['sc2'].replace(0, nan)

    # now extract all other columns and add 'spkitemid'
    other_columns = ['spkitemid'] + [column for column in df_filtered.columns
                     if column not in not_other_columns]
    df_filtered_other_columns = df_filtered[other_columns]

    return (df_filtered_features,
            df_filtered_metadata,
            df_filtered_other_columns,
            df_excluded,
            df_filtered_length,
            df_filtered_human_scores,
            df_responses_with_excluded_flags,
            trim_min,
            trim_max,
            feature_names)


def rename_default_columns(df,
                           requested_feature_names,
                           id_column,
                           first_human_score_column,
                           second_human_score_column,
                           length_column,
                           system_score_column,
                           candidate_column):

    """
    Standardize all column names and rename all columns
    with default names to ##NAME##.
    """

    columns = [id_column,
               first_human_score_column,
               second_human_score_column,
               length_column,
               system_score_column,
               candidate_column]
    defaults = ['spkitemid', 'sc1', 'sc2', 'length', 'raw', 'candidate']

    # create a dictionary of name mapping for used columns
    name_mapping = dict(filter(lambda t: t[0] !=None, zip(columns,
                                                          defaults)))

    # find the columns where the names match the default names
    columns_with_correct_default_names = [column for (column, default) in name_mapping.items()
                                          if column == default]

    # find the columns with default names reserved for other columns
    # which are not used as features in the model
    columns_with_incorrect_default_names = [column for column in df.columns
                                            if (column in defaults and
                                                column not in columns_with_correct_default_names and
                                                column not in requested_feature_names)]
    # rename these columns
    if columns_with_incorrect_default_names:
        new_column_names = ['##{}##'.format(column) for column in columns_with_incorrect_default_names]
        df.rename(columns=dict(zip(columns_with_incorrect_default_names,
                                   new_column_names)),
                  inplace=True)

    # find the columns where the names do not match the default
    columns_with_custom_names = [column for column in name_mapping
                                 if column not in columns_with_correct_default_names]

    # rename the custom-named columns to default values
    for column in columns_with_custom_names:

        # if the column has already been renamed because it used a
        # default name, then use the updated name
        if column in columns_with_incorrect_default_names:
            df.rename(columns={'##{}##'.format(column):
                               name_mapping[column]},
                      inplace=True)
        else:
            df.rename(columns={column:
                               name_mapping[column]},
                      inplace=True)

    return df

