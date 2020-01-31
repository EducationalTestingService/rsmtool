import os
import shutil
import warnings

from glob import glob
from os.path import basename, exists, join

from nose.tools import raises
from parameterized import param, parameterized

from rsmtool.configuration_parser import ConfigurationParser
from rsmtool.test_utils import (check_file_output,
                                check_report,
                                check_scaled_coefficients,
                                check_generated_output,
                                check_run_experiment,
                                do_run_experiment)

# allow test directory to be set via an environment variable
# which is needed for package testing
TEST_DIR = os.environ.get('TESTDIR', None)
if TEST_DIR:
    rsmtool_test_dir = TEST_DIR
else:
    from rsmtool.test_utils import rsmtool_test_dir


@parameterized([
    param('lr-no-standardization', 'lr_no_standardization'),
    param('lr-exclude-test-flags', 'lr_exclude_test_flags'),
    param('lr-exclude-train-and-test-flags', 'lr_exclude_train_and_test_flags'),
    param('lr-with-sas', 'lr_with_sas'),
    param('lr-with-xlsx-output', 'lr_with_xlsx_output', file_format='xlsx'),
    param('lr-with-tsv-output', 'lr_with_tsv_output', file_format='tsv'),
    param('lr-with-thumbnails', 'lr_with_thumbnails'),
    param('lr-with-thumbnails-subgroups', 'lr_with_thumbnails_subgroups', subgroups=['L1']),
    param('lr-with-feature-list', 'lr_with_feature_list'),
    param('lr-with-length-non-numeric', 'lr_with_length_non_numeric'),
    param('lr-with-feature-list-and-transformation', 'lr_with_feature_list_and_transformation'),
    param('lr-with-trim-tolerance', 'lr_with_trim_tolerance'),
    param('lr-subgroups-with-dictionary-threshold-and-empty-group',
          'lr_subgroups_with_dictionary_threshold_and_empty_group',
          subgroups=['L1', 'QUESTION']),
    param('lr-subgroups-with-numeric-threshold-and-empty-group',
          'lr_subgroups_with_numeric_threshold_and_empty_group',
          subgroups=['L1', 'QUESTION']),
    param('lr-subgroups-h2-long-feature-names',
          'lr_subgroups_h2_long_feature_names', subgroups=['L1', 'QUESTION'], consistency=True)
])
def test_run_experiment_parameterized(*args, **kwargs):
    if TEST_DIR:
        kwargs['given_test_dir'] = TEST_DIR
    check_run_experiment(*args, **kwargs)


def test_run_experiment_lr_with_cfg():
    # basic experiment with a LinearRegression model

    source = 'lr-cfg'
    experiment_id = 'lr_cfg'
    config_file = join(rsmtool_test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.cfg'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(rsmtool_test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_with_object():

    # test rsmtool using the Configuration object, rather than a file;
    # we pass the `filepath` attribute after constructing the Configuraiton object
    # to ensure that the results are identical to what we would expect if we had
    # run this test with a configuration file instead.

    source = 'lr-object'
    experiment_id = 'lr_object'

    config_file = join(rsmtool_test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))

    config_dict = {"train_file": "../../files/train.csv",
                   "id_column": "ID",
                   "use_scaled_predictions": True,
                   "test_label_column": "score",
                   "train_label_column": "score",
                   "test_file": "../../files/test.csv",
                   "trim_max": 6,
                   "features": "features.csv",
                   "trim_min": 1,
                   "model": "LinearRegression",
                   "experiment_id": "lr_object",
                   "description": "Using all features with an LinearRegression model."}

    config_parser = ConfigurationParser()
    config_parser.load_config_from_dict(config_dict)
    config_obj = config_parser.normalize_validate_and_process_config()
    config_obj.filepath = config_file

    do_run_experiment(source, experiment_id, config_obj)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(rsmtool_test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report


def test_run_experiment_lr_with_object_no_path():

    # test rsmtool using the Configuration object, rather than a file;
    # we pass the `filepath` attribute after constructing the Configuraiton object
    # to ensure that the results are identical to what we would expect if we had
    # run this test with a configuration file instead.

    source = 'lr-object_no_path'
    experiment_id = 'lr_object_no_path'

    local_dir = 'temp_for_testing_config'
    os.mkdir(local_dir)
    for file in ['train.csv', 'test.csv']:
      shutil.copy(join(rsmtool_test_dir, 'data', 'files', file),
                  join(local_dir, file))
    shutil.copy(join(rsmtool_test_dir, 'data', 'experiments', 
                     'lr-object-no-path', 'features.csv'), 
                join(local_dir, 'features.csv'))

    config_dict = {"train_file": "{}/train.csv".format(local_dir),
                   "id_column": "ID",
                   "use_scaled_predictions": True,
                   "test_label_column": "score",
                   "train_label_column": "score",
                   "test_file": "{}/test.csv".format(local_dir),
                   "trim_max": 6,
                   "features": "{}/features.csv".format(local_dir),
                   "trim_min": 1,
                   "model": "LinearRegression",
                   "experiment_id": "lr_object",
                   "description": "Using all features with an LinearRegression model."}

    config_parser = ConfigurationParser()
    config_parser.load_config_from_dict(config_dict)
    config_obj = config_parser.normalize_validate_and_process_config()

    do_run_experiment(source, experiment_id, config_obj)
    shutil.rmtree(local_dir)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(rsmtool_test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_report, html_report




@raises(ValueError)
def test_run_experiment_duplicate_feature_names():

    # rsmtool experiment with duplicate feature names
    source = 'lr-with-duplicate-feature-names'
    experiment_id = 'lr_with_duplicate_feature_names'
    config_file = join(rsmtool_test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_old_config():
    # basic experiment with a LinearRegression model but using an
    # old style configuration file

    source = 'lr-old-config'
    experiment_id = 'empWt'
    config_file = join(rsmtool_test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))

    # run this experiment but suppress the expected deprecation warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        do_run_experiment(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_feature_json():
    # basic experiment with a LinearRegression model but using
    # feature json file

    source = 'lr-feature-json'
    experiment_id = 'lr'
    config_file = join(rsmtool_test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))

    # run this experiment but suppress the expected deprecation warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        do_run_experiment(source, experiment_id, config_file)


@raises(FileNotFoundError)
def test_run_experiment_wrong_train_file_path():
    # basic experiment with the path in train_file field pointing to
    # a non-existing file
    source = 'lr-wrong-path'
    experiment_id = 'lr'
    config_file = join(rsmtool_test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)


@raises(FileNotFoundError)
def test_run_experiment_wrong_feature_file_path():
    # basic experiment with the path in features field pointing to
    # a non-existing file
    source = 'lr-wrong-path-features'
    experiment_id = 'lr'
    config_file = join(rsmtool_test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)


@raises(ValueError)
def test_run_experiment_lr_length_column_and_feature_list():
    # experiment with feature as list instead of file name
    # and length included in feature list and as length column

    source = 'lr-with-length-and-feature-list'
    experiment_id = 'lr_with_length_and_feature'
    config_file = join(rsmtool_test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)
