from glob import glob
from os.path import basename, dirname, exists, join

from nose.tools import raises
from rsmtool.configuration_parser import ConfigurationParser

from rsmtool.test_utils import (check_file_output,
                                check_report,
                                do_run_summary)

# get the directory containing the tests
test_dir = dirname(__file__)


def test_run_experiment_lr_summary():

    # basic rsmsummarize experiment comparing several rsmtool experiments
    source = 'lr-self-summary'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmsummarize.json')
    do_run_summary(source, config_file)

    html_report = join('test_outputs', source, 'report', 'model_comparison_report.html')

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_file_output, csv_file, expected_csv_file

    yield check_report, html_report


def test_run_experiment_linearsvr_summary():

    # basic rsmsummarize experiment comparing an experiment
    # which uses a SKLL model to itself
    source = 'linearsvr-self-summary'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmsummarize.json')
    do_run_summary(source, config_file)

    html_report = join('test_outputs', source, 'report', 'model_comparison_report.html')

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_file_output, csv_file, expected_csv_file

    yield check_report, html_report


def test_run_experiment_lr_eval_summary():

    # basic rsmsummarize experiment comparing an rsmtool and rsmeval experiments
    source = 'lr-self-eval-summary'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmsummarize.json')
    do_run_summary(source, config_file)

    html_report = join('test_outputs', source, 'report', 'model_comparison_report.html')

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_file_output, csv_file, expected_csv_file

    yield check_report, html_report


def test_run_experiment_lr_summary_with_custom_sections_and_custom_order():

    # basic rsmsummarize experiment comparing a LinearRegression
    # experiment to itself with a custom list of sections
    source = 'lr-self-summary-with-custom-sections'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmsummarize.json')
    do_run_summary(source, config_file)

    html_report = join('test_outputs', source, 'report', 'model_comparison_report.html')

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

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
    config_file = join(test_dir,
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
    config_file = join(test_dir,
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
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmsummarize.json')
    do_run_summary(source, config_file)


def test_run_experiment_lr_summary_with_object():

    # basic rsmsummarize experiment comparing several rsmtool experiments
    source = 'lr-self-summary-object'

    config_file = join(test_dir,
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
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    csv_files = glob(join(output_dir, '*.csv'))
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_file_output, csv_file, expected_csv_file

    yield check_report, html_report


def test_run_experiment_lr_summary_with_tsv_inputs():

    # basic rsmsummarize experiment comparing several rsmtool experiments
    # inputs are TSVs rather than CSVs (outputs are still CSVs)

    source = 'lr-self-summary-with-tsv-inputs'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmsummarize.json')
    do_run_summary(source, config_file)

    html_report = join('test_outputs', source, 'report', 'model_comparison_report.html')

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    tsv_files = glob(join(output_dir, '*.tsv'))
    for tsv_file in tsv_files:
        tsv_filename = basename(tsv_file)
        expected_tsv_file = join(expected_output_dir, tsv_filename)

        if exists(expected_tsv_file):
            yield check_file_output, tsv_file, expected_tsv_file

    yield check_report, html_report
