import os

from glob import glob
from os.path import basename, exists, join

from nose.tools import raises
from parameterized import param, parameterized

from rsmtool.configuration_parser import ConfigurationParser

from rsmtool.test_utils import (check_file_output,
                                check_report,
                                check_run_summary,
                                do_run_summary)

# allow test directory to be set via an environment variable
# which is needed for package testing
TEST_DIR = os.environ.get('TESTDIR', None)
if TEST_DIR:
    rsmtool_test_dir = TEST_DIR
else:
    from rsmtool.test_utils import rsmtool_test_dir


def setup_func():
    global DIRS_TO_REMOVE
    DIRS_TO_REMOVE = []


def teardown_func():
    for d in DIRS_TO_REMOVE:
        if exists(d):
            shutil.rmtree(d)



@parameterized([
    param('lr-self-summary'),
    param('linearsvr-self-summary'),
    param('lr-self-eval-summary'),
    param('lr-self-summary-with-custom-sections'),
    param('lr-self-summary-with-tsv-inputs'),
    param('lr-self-summary-with-tsv-output', file_format='tsv'),
    param('lr-self-summary-with-xlsx-output', file_format='xlsx'),
    param('lr-self-summary-no-scaling'),
    param('lr-self-summary-with-h2'),
    param('summary-with-custom-names'),
    param('lr-self-summary-null-trim-min')
])
def test_run_experiment_parameterized(*args, **kwargs):
    if TEST_DIR:
        kwargs['given_test_dir'] = TEST_DIR
    check_run_summary(*args, **kwargs)


def test_run_experiment_lr_summary_with_object():

    # test rsmsummarize using the Configuration object, rather than a file;
    # we pass the `filepath` attribute after constructing the Configuraiton object
    # to ensure that the results are identical to what we would expect if we had
    # run this test with a configuration file instead.
    source = 'lr-self-summary-object'

    config_file = join(rsmtool_test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmsummarize.json')

    config_dict = {"summary_id": "model_comparison",
                   "experiment_dirs": ["lr-subgroups", "lr-subgroups", "lr-subgroups"],
                   "description": "Comparison of rsmtool experiment with itself."}

    config_parser = ConfigurationParser()
    config_parser.load_config_from_dict(config_dict)
    config_obj = config_parser.normalize_validate_and_process_config(context='rsmsummarize')
    config_obj.filepath = config_file

    do_run_summary(source, config_obj)

    html_report = join('test_outputs', source, 'report', 'model_comparison_report.html')

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(rsmtool_test_dir, 'data', 'experiments', source, 'output')

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_file_output, csv_file, expected_csv_file

    yield check_report, html_report


def test_run_experiment_lr_summary_no_trim():

    # experiment to check the condition where no trim values can be located
    # also uses the `Configuration` object directly
    source = 'lr-self-summary-no-trim'

    config_file = join(rsmtool_test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmsummarize.json')

    config_dict = {"summary_id": "model_comparison",
                   "experiment_dirs": ["lr-subgroups1", "lr-subgroups2", "lr-subgroups3"],
                   "description": "Comparison of rsmtool without trim values"}

    config_parser = ConfigurationParser()
    config_parser.load_config_from_dict(config_dict)
    config_obj = config_parser.normalize_validate_and_process_config(context='rsmsummarize')
    config_obj.filepath = config_file

    do_run_summary(source, config_obj)

    html_report = join('test_outputs', source, 'report', 'model_comparison_report.html')

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(rsmtool_test_dir, 'data', 'experiments', source, 'output')

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_file_output, csv_file, expected_csv_file

    yield check_report, html_report


@raises(FileNotFoundError)
def test_run_experiment_summary_wrong_directory():

    # rsmsummarize experiment where the specified directory
    # does not exist
    source = 'summary-wrong-directory'
    config_file = join(rsmtool_test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmsummarize.json')
    do_run_summary(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_summary_no_csv_directory():

    # rsmsummarize experiment where the specified directory
    # does not contain any rsmtool experiments
    source = 'summary-no-output-dir'
    config_file = join(rsmtool_test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmsummarize.json')
    do_run_summary(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_summary_no_json():

    # rsmsummarize experiment where the specified directory
    # does not contain any json files
    source = 'summary-no-json-file'
    config_file = join(rsmtool_test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmsummarize.json')
    do_run_summary(source, config_file)


@raises(ValueError)
def test_run_experiment_summary_too_many_jsons():

    # rsmsummarize experiment where the specified directory
    # does contains several jsons files and the user
    # specified experiment names
    source = 'summary-too-many-jsons'
    config_file = join(rsmtool_test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmsummarize.json')
    do_run_summary(source, config_file)
