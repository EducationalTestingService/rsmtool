import os
import tempfile

from glob import glob
from os import getcwd
from os.path import basename, exists, join

from nose.tools import raises
from parameterized import param, parameterized

from rsmtool import compute_and_save_predictions

from rsmtool.configuration_parser import ConfigurationParser
from rsmtool.test_utils import (check_file_output,
                                check_report,
                                check_scaled_coefficients,
                                check_generated_output,
                                check_run_prediction,
                                copy_data_files,
                                do_run_experiment,
                                do_run_prediction)

# allow test directory to be set via an environment variable
# which is needed for package testing
TEST_DIR = os.environ.get('TESTDIR', None)
if TEST_DIR:
    rsmtool_test_dir = TEST_DIR
else:
    from rsmtool.test_utils import rsmtool_test_dir


@parameterized([
    param('lr-predict'),
    param('lr-predict-with-score'),
    param('lr-predict-missing-values', excluded=True),
    param('lr-predict-with-subgroups'),
    param('lr-predict-with-candidate'),
    param('lr-predict-illegal-transformations', excluded=True),
    param('lr-predict-tsv-input-files'),
    param('lr-predict-xlsx-input-files'),
    param('lr-predict-jsonlines-input-files'),
    param('lr-predict-nested-jsonlines-input-files'),
    param('lr-predict-no-standardization'),
    param('lr-predict-with-tsv-output', file_format='tsv'),
    param('lr-predict-with-xlsx-output', file_format='xlsx'),
    param('logistic-regression-predict'),
    param('logistic-regression-predict-expected-scores'),
    param('svc-predict-expected-scores'),
    param('lr-predict-with-custom-tolerance'),
    param('lr-predict-no-tolerance')
])
def test_run_experiment_parameterized(*args, **kwargs):
    if TEST_DIR:
        kwargs['given_test_dir'] = TEST_DIR
    check_run_prediction(*args, **kwargs)


def test_run_experiment_lr_rsmtool_and_rsmpredict():
    '''
    this test is to make sure that both rsmtool
    and rsmpredict generate the same files
    '''

    source = 'lr-rsmtool-rsmpredict'
    experiment_id = 'lr_rsmtool_rsmpredict'
    rsmtool_config_file = join(rsmtool_test_dir,
                               'data',
                               'experiments',
                               source,
                               '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, rsmtool_config_file)
    rsmpredict_config_file = join(rsmtool_test_dir,
                                  'data',
                                  'experiments',
                                  source,
                                  'rsmpredict.json')
    do_run_prediction(source, rsmpredict_config_file)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(rsmtool_test_dir, 'data', 'experiments', source, 'output')
    csv_files = glob(join(output_dir, '*.csv'))
    html_report = join('test_outputs', source, 'report', '{}_report.html'.format(experiment_id))

    # Check the results for  rsmtool
    for csv_file in csv_files:
        csv_filename = basename(csv_file)
        expected_csv_file = join(expected_output_dir, csv_filename)

        if exists(expected_csv_file):
            yield check_file_output, csv_file, expected_csv_file

    yield check_scaled_coefficients, source, experiment_id
    yield check_generated_output, csv_files, experiment_id, 'rsmtool'
    yield check_report, html_report

    # check that the rsmpredict generated the same results
    for csv_pair in [('predictions.csv',
                      '{}_pred_processed.csv'.format(experiment_id)),
                     ('preprocessed_features.csv',
                      '{}_test_preprocessed_features.csv'.format(experiment_id))]:
        output_file = join(output_dir, csv_pair[0])
        expected_output_file = join(expected_output_dir, csv_pair[1])

        yield check_file_output, output_file, expected_output_file


def test_run_experiment_lr_predict_with_object():
    '''
    test rsmpredict using the Configuration object, rather than a file
    '''

    source = 'lr-predict-object'

    configdir = join(rsmtool_test_dir,
                     'data',
                     'experiments',
                     source)

    config_dict = {"id_column": "ID",
                   "input_features_file": "../../files/test.csv",
                   "experiment_dir": "existing_experiment",
                   "experiment_id": "lr"
                   }

    config_parser = ConfigurationParser()
    config_parser.load_config_from_dict(config_dict,
                                        configdir=configdir)
    config_obj = config_parser.normalize_validate_and_process_config(context='rsmpredict')

    check_run_prediction(source,
                         given_test_dir=rsmtool_test_dir,
                         config_obj_or_dict=config_obj)


def test_run_experiment_lr_predict_with_dictionary():
    '''
    test rsmpredict using the dictionary object, rather than a file
    '''

    source = 'lr-predict-dict'

    # set up a temporary directory since
    # we will be using getcwd
    temp_dir = tempfile.TemporaryDirectory(prefix=getcwd())

    old_file_dict = {'feature_file': 'data/files/test.csv',
                     'experiment_dir': 'data/experiments/lr-predict-dict/existing_experiment'}

    new_file_dict = copy_data_files(temp_dir.name,
                                    old_file_dict,
                                    rsmtool_test_dir)

    config_dict = {"id_column": "ID",
                   "input_features_file": new_file_dict['feature_file'],
                   "experiment_dir": new_file_dict['experiment_dir'],
                   "experiment_id": "lr"}

    check_run_prediction(source,
                         given_test_dir=rsmtool_test_dir,
                         config_obj_or_dict=config_dict)


@raises(ValueError)
def test_run_experiment_lr_predict_with_repeated_ids():

    # rsmpredict experiment with non-unique ids
    source = 'lr-predict-with-repeated-ids'
    config_file = join(rsmtool_test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(ValueError)
def test_compute_predictions_wrong_input_format():
    config_list = [('experiment_id', 'AAAA'),
                   ('train_file', 'some_path')]
    with tempfile.TemporaryDirectory() as temp_dir:
        compute_and_save_predictions(config_list, temp_dir)


@raises(FileNotFoundError)
def test_run_experiment_lr_predict_missing_model_file():
    '''
    rsmpredict experiment with missing model file
    '''
    source = 'lr-predict-missing-model-file'
    config_file = join(rsmtool_test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_lr_predict_missing_feature_file():
    '''
    rsmpredict experiment with missing feature file
    '''
    source = 'lr-predict-missing-feature-file'
    config_file = join(rsmtool_test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_lr_predict_missing_postprocessing_file():
    '''
    rsmpredict experiment with missing post-processing file
    '''
    source = 'lr-predict-missing-postprocessing-file'
    config_file = join(rsmtool_test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_predict_no_input_feature_file():
    '''
    rsmpredict experiment with missing feature file
    '''
    source = 'lr-predict-no-input-feature-file'
    config_file = join(rsmtool_test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_predict_no_experiment_dir():
    '''
    rsmpredict experiment with missing experiment dir
    '''
    source = 'lr-predict-no-experiment-dir'
    config_file = join(rsmtool_test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_predict_no_output_dir():
    '''
    rsmpredict experiment where experiment_dir
    does not containt output directory
    '''
    source = 'lr-predict-no-output-dir'
    config_file = join(rsmtool_test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_predict_no_experiment_id():
    '''
    rsmpredict experiment ehere the experiment_dir
    does not contain the experiment with the stated id
    '''
    source = 'lr-predict-no-experiment-id'
    config_file = join(rsmtool_test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(KeyError)
def test_run_experiment_lr_predict_missing_columns():
    '''
    rsmpredict experiment with missing columns
    from the config file
    '''
    source = 'lr-predict-missing-columns'
    config_file = join(rsmtool_test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(KeyError)
def test_run_experiment_lr_predict_missing_feature():
    '''
    rsmpredict experiment with missing features
    '''
    source = 'lr-predict-missing-feature'
    config_file = join(rsmtool_test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(ValueError)
def test_run_experiment_lr_predict_no_numeric_feature_values():
    '''
    rsmpredict experiment with missing post-processing file
    '''
    source = 'lr-predict-no-numeric-feature-values'
    config_file = join(rsmtool_test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(ValueError)
def test_run_experiment_predict_expected_scores_builtin_model():
    '''
    rsmpredict experiment for expected scores but with
    a built-in model which is not supporte
    '''
    source = 'lr-predict-expected-scores-builtin-model'
    config_file = join(rsmtool_test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(ValueError)
def test_run_experiment_predict_expected_scores_wrong_skll_model():
    '''
    rsmpredict experiment for expected scores but with
    a non-probabilistic SKLL learner
    '''
    source = 'predict-expected-scores-wrong-skll-model'
    config_file = join(rsmtool_test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(ValueError)
def test_run_experiment_predict_expected_scores_non_probablistic_svc():
    '''
    rsmpredict experiment for expected scores but with
    a non-probabilistic SKLL learner
    '''
    source = 'predict-expected-scores-non-probabilistic-svc'
    config_file = join(rsmtool_test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)
