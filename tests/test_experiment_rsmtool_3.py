import warnings

from glob import glob
from os.path import basename, dirname, exists, join

from nose.tools import raises
from rsmtool.configuration_parser import ConfigurationParser

from rsmtool.test_utils import (check_file_output,
                                check_report,
                                check_scaled_coefficients,
                                check_subgroup_outputs,
                                check_generated_output,
                                do_run_experiment)

# get the directory containing the tests
test_dir = dirname(__file__)


@raises(ValueError)
def test_run_experiment_duplicate_feature_names():

    # rsmtool experiment with duplicate feature names
    source = 'lr-with-duplicate-feature-names'
    experiment_id = 'lr_with_duplicate_feature_names'
    config_file = join(test_dir,
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
    config_file = join(test_dir,
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
    config_file = join(test_dir,
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
    config_file = join(test_dir,
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
    config_file = join(test_dir,
                       'data',
                       'experiments',
                        source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)



def test_run_experiment_lr_with_cfg():
    # basic experiment with a LinearRegression model

    source = 'lr-cfg'
    experiment_id = 'lr_cfg'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.cfg'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
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
    # basic experiment with a LinearRegression model

    source = 'lr-object'
    experiment_id = 'lr_object'

    config_file = join(test_dir,
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
    config_obj = config_file

    do_run_experiment(source, experiment_id, config_obj)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
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


def test_run_experiment_lr_no_standardization():
    # basic experiment with a LinearRegression model,
    # with no standardization of features

    source = 'lr-no-standardization'
    experiment_id = 'lr_no_standardization'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))

    do_run_experiment(source, experiment_id, config_file)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
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


def test_run_experiment_lr_exclude_test_flags():
    # basic experiment with a LinearRegression model,
    # with only test flags removed

    source = 'lr-exclude-test-flags'
    experiment_id = 'lr_exclude_test_flags'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
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


def test_run_experiment_lr_exclude_train_and_test_flags():
    # basic experiment with a LinearRegression model,
    # with different train and test flags

    source = 'lr-exclude-train-and-test-flags'
    experiment_id = 'lr_exclude_train_and_test_flags'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
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


def test_run_experiment_lr_with_xlsx_output():
    # basic experiment with a LinearRegression model
    # with XLSX files as outputs

    source = 'lr-with-xlsx-output'
    experiment_id = 'lr'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    xlsx_files = glob(join(output_dir, '*.xlsx'))
    for xlsx_file in xlsx_files:
        xlsx_filename = basename(xlsx_file)
        expected_xlsx_file = join(expected_output_dir, xlsx_filename)

        if exists(expected_xlsx_file):
            yield check_file_output, xlsx_file, expected_xlsx_file, 'xlsx'

    yield check_generated_output, xlsx_files, experiment_id, 'rsmtool', 'xlsx'
    yield check_scaled_coefficients, source, experiment_id, 'xlsx'
    yield check_report, html_report


def test_run_experiment_lr_with_tsv_output():
    # basic experiment with a LinearRegression model
    # with TSV files as outputs

    source = 'lr-with-tsv-output'
    experiment_id = 'lr'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    tsv_files = glob(join(output_dir, '*.tsv'))
    for tsv_file in tsv_files:
        tsv_filename = basename(tsv_file)
        expected_tsv_file = join(expected_output_dir, tsv_filename)

        if exists(expected_tsv_file):
            yield check_file_output, tsv_file, expected_tsv_file, 'tsv'

    yield check_generated_output, tsv_files, experiment_id, 'rsmtool', 'tsv'
    yield check_scaled_coefficients, source, experiment_id, 'tsv'
    yield check_report, html_report


def test_run_experiment_lr_with_thumbnails():
    # basic experiment with a LinearRegression model
    # with thumbnail conversion

    source = 'lr-with-thumbnails'
    experiment_id = 'lr'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))

    do_run_experiment(source, experiment_id, config_file)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
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


def test_run_experiment_lr_with_thumbnails_subgroups():

    # basic experiment with LinearRegression model but also including
    # subgroup analyses, with thumbnail conversion

    source = 'lr-with-thumbnails-subgroups'
    experiment_id = 'lr_subgroups'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_file_output, csv_file, expected_csv_file

    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
    yield check_scaled_coefficients, source, experiment_id
    yield check_subgroup_outputs, output_dir, experiment_id, ['L1']
    yield check_report, html_report


def test_run_experiment_lr_with_feature_list():
    # basic experiment with a LinearRegression model
    # with feature as list instead of file name

    source = 'lr-with-feature-list'
    experiment_id = 'lr'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
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
def test_run_experiment_lr_length_column_and_feature_list():
    # experiment with feature as list instead of file name
    # and length included in feature list and as length column

    source = 'lr-with-length-and-feature-list'
    experiment_id = 'lr_with_length_and_feature'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)


def test_run_experiment_lr_with_feature_list_and_transformation():
    # basic experiment with a LinearRegression model
    # with feature as list instead of file name

    source = 'lr-with-feature-list-and-transformation'
    experiment_id = 'lr'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, config_file)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
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
