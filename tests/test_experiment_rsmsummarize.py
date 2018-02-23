from glob import glob
from os.path import basename, exists, join

from nose.tools import raises
from parameterized import param, parameterized

from rsmtool.configuration_parser import ConfigurationParser

from rsmtool.test_utils import (check_file_output,
                                check_report,
                                check_run_summary,
                                do_run_summary,
                                rsmtool_test_dir)


@parameterized([
    param('lr-self-summary'),
    param('linearsvr-self-summary'),
    param('lr-self-eval-summary'),
    param('lr-self-summary-with-custom-sections'),
    param('lr-self-summary-with-tsv-inputs'),
    param('lr-self-summary-with-tsv-output', file_format='tsv'),
    param('lr-self-summary-with-xlsx-output', file_format='xlsx'),
    param('lr-self-summary-no-scaling')
])
def test_run_experiment_parameterized(*args, **kwargs):
    check_run_summary(*args, **kwargs)


def test_run_experiment_lr_summary_with_object():

    # basic rsmsummarize experiment comparing several rsmtool experiments
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
    config_obj = config_file

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
