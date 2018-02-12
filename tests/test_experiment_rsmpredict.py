from glob import glob
from os.path import basename, dirname, exists, join

from nose.tools import raises

from rsmtool.test_utils import (check_file_output,
                                check_report,
                                check_scaled_coefficients,
                                check_generated_output,
                                do_run_experiment,
                                do_run_prediction)

# get the directory containing the tests
test_dir = dirname(__file__)


def test_run_experiment_lr_predict():

    # basic experiment using rsmpredict

    source = 'lr-predict'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    for csv_file in ['predictions.csv', 'preprocessed_features.csv']:
        output_file = join(output_dir, csv_file)
        expected_output_file = join(expected_output_dir, csv_file)

        yield check_file_output, output_file, expected_output_file


def test_run_experiment_lr_predict_with_score():

    # rsmpredict experiment with human score

    source = 'lr-predict-with-score'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    for csv_file in ['predictions.csv', 'preprocessed_features.csv']:
        output_file = join(output_dir, csv_file)
        expected_output_file = join(expected_output_dir, csv_file)

        yield check_file_output, output_file, expected_output_file


def test_run_experiment_lr_predict_missing_values():

    # basic experiment using rsmpredict when the supplied feature file
    # contains reponses with non-numeric feature values

    source = 'lr-predict-missing-values'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    for csv_file in ['predictions.csv', 'predictions_excluded_responses.csv',
                     'preprocessed_features.csv']:
        output_file = join(output_dir, csv_file)
        expected_output_file = join(expected_output_dir, csv_file)

        yield check_file_output, output_file, expected_output_file


def test_run_experiment_lr_predict_with_subgroups():

    # basic experiment using rsmpredict with subgroups and other columns

    source = 'lr-predict-with-subgroups'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    for csv_file in ['predictions.csv', 'preprocessed_features.csv']:
        output_file = join(output_dir, csv_file)
        expected_output_file = join(expected_output_dir, csv_file)

        yield check_file_output, output_file, expected_output_file


def test_run_experiment_lr_predict_with_candidate():

    # basic experiment using rsmpredict with candidate column

    source = 'lr-predict-with-candidate'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    for csv_file in ['predictions.csv', 'preprocessed_features.csv']:
        output_file = join(output_dir, csv_file)
        expected_output_file = join(expected_output_dir, csv_file)

        yield check_file_output, output_file, expected_output_file


def test_run_experiment_lr_predict_illegal_transformations():

    # rsmpredict experiment where the transformations applied to
    # the new data lead to inf or NaN values. This responses should
    # be treated as if the feature values are missing.

    source = 'lr-predict-illegal-transformations'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    for csv_file in ['predictions.csv', 'predictions_excluded_responses.csv',
                     'preprocessed_features.csv']:
        output_file = join(output_dir, csv_file)
        expected_output_file = join(expected_output_dir, csv_file)

        yield check_file_output, output_file, expected_output_file


def test_run_experiment_lr_rsmtool_and_rsmpredict():

    # this test is to make sure that both rsmtool
    # and rsmpredict generate the same files

    source = 'lr-rsmtool-rsmpredict'
    experiment_id = 'lr_rsmtool_rsmpredict'
    rsmtool_config_file = join(test_dir,
                               'data',
                               'experiments',
                               source,
                               '{}.json'.format(experiment_id))
    do_run_experiment(source, experiment_id, rsmtool_config_file)
    rsmpredict_config_file = join(test_dir,
                                  'data',
                                  'experiments',
                                  source,
                                  'rsmpredict.json')
    do_run_prediction(source, rsmpredict_config_file)
    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')
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


def test_run_experiment_lr_predict_tsv_input_files():

    # rsmpredict experiment with input file in .tsv format

    source = 'lr-predict-tsv-input-files'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    for csv_file in ['predictions.csv', 'preprocessed_features.csv']:
        output_file = join(output_dir, csv_file)
        expected_output_file = join(expected_output_dir, csv_file)

        yield check_file_output, output_file, expected_output_file


def test_run_experiment_lr_predict_xlsx_input_files():

    # rsmpredict experiment with input file in .xlsx format

    source = 'lr-predict-xlsx-input-files'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    for csv_file in ['predictions.csv', 'preprocessed_features.csv']:
        output_file = join(output_dir, csv_file)
        expected_output_file = join(expected_output_dir, csv_file)

        yield check_file_output, output_file, expected_output_file


@raises(ValueError)
def test_run_experiment_lr_predict_with_repeated_ids():

    # rsmpredict experiment with non-unique ids
    source = 'lr-predict-with-repeated-ids'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_lr_predict_missing_model_file():

    # rsmpredict experiment with missing model file
    source = 'lr-predict-missing-model-file'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_lr_predict_missing_feature_file():

    # rsmpredict experiment with missing feature file
    source = 'lr-predict-missing-feature-file'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_lr_predict_missing_postprocessing_file():

    # rsmpredict experiment with missing post-processing file
    source = 'lr-predict-missing-postprocessing-file'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_predict_no_input_feature_file():

    # rsmpredict experiment with missing feature file
    source = 'lr-predict-no-input-feature-file'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_predict_no_experiment_dir():

    # rsmpredict experiment with missing experiment dir
    source = 'lr-predict-no-experiment-dir'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_predict_no_output_dir():

    # rsmpredict experiment where experiment_dir
    # does not containt output directory
    source = 'lr-predict-no-output-dir'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(FileNotFoundError)
def test_run_experiment_predict_no_experiment_id():

    # rsmpredict experiment ehere the experiment_dir
    # does not contain the experiment with the stated id
    source = 'lr-predict-no-experiment-id'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(KeyError)
def test_run_experiment_lr_predict_missing_columns():

    # rsmpredict experiment with missing columns
    # from the config file
    source = 'lr-predict-missing-columns'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(KeyError)
def test_run_experiment_lr_predict_missing_feature():

    # rsmpredict experiment with missing features
    source = 'lr-predict-missing-feature'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(ValueError)
def test_run_experiment_lr_predict_no_numeric_feature_values():

    # rsmpredict experiment with missing post-processing file
    source = 'lr-predict-no-numeric-feature-values'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


def test_run_experiment_lr_predict_no_standardization():

    # rsmpredict experiment with no standardization of features

    source = 'lr-predict-no-standardization'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')

    do_run_prediction(source, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    for csv_file in ['predictions.csv', 'preprocessed_features.csv']:
        output_file = join(output_dir, csv_file)
        expected_output_file = join(expected_output_dir, csv_file)

        yield check_file_output, output_file, expected_output_file


def test_run_experiment_lr_predict_with_tsv_output():

    # basic experiment using rsmpredict
    # output in TSV format

    source = 'lr-predict-with-tsv-output'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    for tsv_file in ['predictions.tsv', 'preprocessed_features.tsv']:
        output_file = join(output_dir, tsv_file)
        expected_output_file = join(expected_output_dir, tsv_file)

        yield check_file_output, output_file, expected_output_file, 'tsv'


def test_run_experiment_lr_predict_with_xlsx_output():

    # basic experiment using rsmpredict
    # output in TSV format

    source = 'lr-predict-with-xlsx-output'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    for xlsx_file in ['predictions.xlsx', 'preprocessed_features.xlsx']:
        output_file = join(output_dir, xlsx_file)
        expected_output_file = join(expected_output_dir, xlsx_file)

        yield check_file_output, output_file, expected_output_file, 'xlsx'


def test_run_experiment_logistic_regression_predict():

    # basic experiment using rsmpredict with logistic regression model

    source = 'logistic-regression-predict'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    for csv_file in ['predictions.csv', 'preprocessed_features.csv']:
        output_file = join(output_dir, csv_file)
        expected_output_file = join(expected_output_dir, csv_file)

        yield check_file_output, output_file, expected_output_file


def test_run_experiment_logistic_regression_predict_expected_scores():

    # basic experiment using rsmpredict with logistic regression and expected scores

    source = 'logistic-regression-predict-expected-scores'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    for csv_file in ['predictions.csv', 'preprocessed_features.csv']:
        output_file = join(output_dir, csv_file)
        expected_output_file = join(expected_output_dir, csv_file)

        yield check_file_output, output_file, expected_output_file


def test_run_experiment_svc_predict_expected_scores():

    # basic experiment using rsmpredict with svc and expected scores

    source = 'svc-predict-expected-scores'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)

    output_dir = join('test_outputs', source, 'output')
    expected_output_dir = join(test_dir, 'data', 'experiments', source, 'output')

    for csv_file in ['predictions.csv', 'preprocessed_features.csv']:
        output_file = join(output_dir, csv_file)
        expected_output_file = join(expected_output_dir, csv_file)

        yield check_file_output, output_file, expected_output_file


@raises(ValueError)
def test_run_experiment_predict_expected_scores_builtin_model():

    # rsmpredict experiment for expected scores but with
    # a built-in model which is not supporte
    source = 'lr-predict-expected-scores-builtin-model'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(ValueError)
def test_run_experiment_predict_expected_scores_wrong_skll_model():

    # rsmpredict experiment for expected scores but with
    # a non-probabilistic SKLL learner
    source = 'predict-expected-scores-wrong-skll-model'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)


@raises(ValueError)
def test_run_experiment_predict_expected_scores_non_probablistic_svc():

    # rsmpredict experiment for expected scores but with
    # a non-probabilistic SKLL learner
    source = 'predict-expected-scores-non-probabilistic-svc'
    config_file = join(test_dir,
                       'data',
                       'experiments',
                       source,
                       'rsmpredict.json')
    do_run_prediction(source, config_file)
