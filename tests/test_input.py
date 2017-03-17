import os
import tempfile
import warnings

from os.path import dirname, join

import pandas as pd

from nose.tools import assert_equal, assert_raises, eq_, ok_, raises
from numpy.testing import assert_array_equal
from pandas.util.testing import assert_frame_equal

from rsmtool.input import (check_flag_column,
                           check_subgroups,
                           check_feature_subset_file,
                           normalize_json_fields,
                           normalize_and_validate_json_feature_file,
                           process_json_fields,
                           read_data_file,
                           rename_default_columns,
                           select_candidates_with_N_or_more_items,
                           validate_and_populate_json_fields,
                           validate_feature_specs)

from rsmtool.convert_feature_json import convert_feature_json_file

_MY_DIR = dirname(__file__)


def check_read_data_file(extension):
    """
    Test whether the ``read_data_file()`` function works as expected.
    """
    df_expected = pd.DataFrame({'id': ['001', '002', '003'],
                                'feature1': [1, 2, 3],
                                'feature2': [4, 5, 6],
                                'gender': ['M', 'F', 'F'],
                                'candidate': ['123', '456', '78901']})

    tempf = tempfile.NamedTemporaryFile(mode='w',
                                        suffix='.{}'.format(extension),
                                        delete=False)
    if extension.lower() == 'csv':
        df_expected.to_csv(tempf, index=False)
    elif extension.lower() == 'tsv':
        df_expected.to_csv(tempf, sep='\t', index=False)
    elif extension.lower() in ['xls', 'xlsx']:
        df_expected.to_excel(tempf.name, index=False)
    tempf.close()

    # now read in the file using `read_data_file()`
    df_read = read_data_file(tempf.name,
                             converters={'id': str, 'candidate': str})

    # get rid of the file now that have read it into memory
    os.unlink(tempf.name)

    assert_frame_equal(df_expected, df_read)


def test_read_data_file():
    # note that we cannot check for capital .xls and .xlsx
    # because xlwt does not support these extensions
    for extension in ['csv', 'tsv', 'xls', 'xlsx', 'CSV', 'TSV']:
        yield check_read_data_file, extension

@raises(ValueError)
def test_read_data_file_wrong_extension():
    check_read_data_file('txt')


def test_normalize_fields():
    data = {'expID': 'experiment_1',
            'train': 'data/rsmtool_smTrain.csv',
            'LRmodel': 'empWt',
            'feature': 'feature/feature_list.json',
            'description': 'A sample model with 9 features '
                           'trained using average score and tested using r1.',
            'test': 'data/rsmtool_smEval.csv',
            'train.lab': 'sc1',
            'crossvalidate': 'yes',
            'test.lab': 'r1',
            'scale': 'scale'}

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        newdata = normalize_json_fields(data)
        ok_('experiment_id' in newdata.keys())
        assert_equal(newdata['experiment_id'], 'experiment_1')
        assert_equal(newdata['use_scaled_predictions'], True)

    # test for non-standard scaling value
    data = {'expID': 'experiment_1',
            'train': 'data/rsmtool_smTrain.csv',
            'LRmodel': 'LinearRegression',
            'scale': 'Yes'}
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        assert_raises(ValueError, normalize_json_fields, data)

    # test when no scaling is specified
    data = {'expID': 'experiment_1',
            'train': 'data/rsmtool_smTrain.csv',
            'LRmodel': 'LinearRegression',
            'feature': 'feature/feature_list.json',
            'description': 'A sample model with 9 features '
                           'trained using average score and tested using r1.',
            'test': 'data/rsmtool_smEval.csv',
            'train.lab': 'sc1',
            'crossvalidate': 'yes',
            'test.lab': 'r1'}

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=DeprecationWarning)
        newdata = normalize_json_fields(data)
        ok_('use_scaled_predictions' not in newdata.keys())


@raises(ValueError)
def test_validate_and_populate_missing_fields():
    data = {'expID': 'test'}
    validate_and_populate_json_fields(data)


@raises(ValueError)
def test_validate_and_populate_min_responses_but_no_candidate():
    data = {'experiment_id': 'experiment_1',
            'train_file': 'data/rsmtool_smTrain.csv',
            'test_file': 'data/rsmtool_smEval.csv',
            'model': 'LinearRegression',
            'min_responses_per_candidate': 5}
    validate_and_populate_json_fields(data)



def test_validate_and_populate_unspecified_fields():
    data = {'experiment_id': 'experiment_1',
            'train_file': 'data/rsmtool_smTrain.csv',
            'test_file': 'data/rsmtool_smEval.csv',
            'model': 'LinearRegression'}
    newdata = validate_and_populate_json_fields(data)
    assert_equal(newdata['id_column'], 'spkitemid')
    assert_equal(newdata['use_scaled_predictions'], False)
    assert_equal(newdata['select_transformations'], False)
    assert_equal(newdata['general_sections'], 'all')
    assert_equal(newdata['description'], '')


@raises(ValueError)
def test_validate_experiment_id_1():
    data = {'experiment_id': 'test experiment',
            'train_file': 'data/rsmtool_smTrain.csv',
            'test_file': 'data/rsmtool_smEval.csv',
            'model': 'LinearRegression'}
    validate_and_populate_json_fields(data)


@raises(ValueError)
def test_validate_experiment_id_2():
    data = {'experiment_id': 'test experiment',
            'predictions_file': 'data/foo',
            'system_score_column': 'h1',
            'trim_min': 1,
            'trim_max': 5}
    validate_and_populate_json_fields(data, context='rsmeval')


@raises(ValueError)
def test_validate_experiment_id_3():
    data = {'comparison_id': 'old vs new',
            'experiment_id_old': 'old_experiment',
            'experiment_dir_old': 'data/old',
            'experiment_id_new': 'new_experiment',
            'experiment_dir_new': 'data/new',}
    validate_and_populate_json_fields(data, context='rsmcompare')


@raises(ValueError)
def test_validate_experiment_id_4():
    data = {'comparison_id': 'old vs new',
            'experiment_id_old': 'old experiment',
            'experiment_dir_old': 'data/old',
            'experiment_id_new': 'new_experiment',
            'experiment_dir_new': 'data/new',}
    validate_and_populate_json_fields(data, context='rsmcompare')


@raises(ValueError)
def test_validate_experiment_id_5():
    data = {'experiment_id': 'this_is_a_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_long_id',
            'train_file': 'data/rsmtool_smTrain.csv',
            'test_file': 'data/rsmtool_smEval.csv',
            'model': 'LinearRegression'}
    validate_and_populate_json_fields(data)


@raises(ValueError)
def test_validate_experiment_id_6():
    data = {'experiment_id': 'this is a really really really really really really really really really really really really really really really really really really really really really really really really really really really long id',
            'predictions_file': 'data/foo',
            'system_score_column': 'h1',
            'trim_min': 1,
            'trim_max': 5}
    validate_and_populate_json_fields(data, context='rsmeval')


@raises(ValueError)
def test_validate_experiment_id_7():
    data = {'comparison_id': 'this_is_a_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_long_id',
            'experiment_id_old': 'old_experiment',
            'experiment_dir_old': 'data/old',
            'experiment_id_new': 'new_experiment',
            'experiment_dir_new': 'data/new',}
    validate_and_populate_json_fields(data, context='rsmcompare')


@raises(ValueError)
def test_validate_experiment_id_8():
    data = {'summary_id': 'model summary',
            'experiment_dirs': []}
    validate_and_populate_json_fields(data, context='rsmsummarize')


@raises(ValueError)
def test_validate_experiment_id_9():
    data = {'summary_id': 'this_is_a_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_really_long_id',
            'experiment_dirs': []}
    validate_and_populate_json_fields(data, context='rsmsummarize')


@raises(ValueError)
def test_validate_and_populate_unknown_fields():
    data = {'experiment_id': 'experiment_1',
            'train_file': 'data/rsmtool_smTrain.csv',
            'test_file': 'data/rsmtool_smEval.csv',
            'description': 'Test',
            'model': 'LinearRegression',
            'output': 'foobar'}
    validate_and_populate_json_fields(data)


def test_process_fields():
    data = {'experiment_id': 'experiment_1',
            'train_file': 'data/rsmtool_smTrain.csv',
            'test_file': 'data/rsmtool_smEval.csv',
            'description': 'Test',
            'model': 'empWt',
            'use_scaled_predictions': 'True',
            'feature_prefix': '1gram, 2gram',
            'subgroups': 'native language, GPA_range',
            'exclude_zero_scores': 'false'}

    newdata = process_json_fields(validate_and_populate_json_fields(data))
    assert_array_equal(newdata['feature_prefix'], ['1gram', '2gram'])
    assert_array_equal(newdata['subgroups'], ['native language', 'GPA_range'])
    eq_(type(newdata['use_scaled_predictions']), bool)
    eq_(newdata['use_scaled_predictions'], True)
    eq_(newdata['exclude_zero_scores'], False)


@raises(ValueError)
def test_process_fields_with_non_boolean():
    data = {'experiment_id': 'experiment_1',
            'train_file': 'data/rsmtool_smTrain.csv',
            'test_file': 'data/rsmtool_smEval.csv',
            'description': 'Test',
            'model': 'empWt',
            'use_scaled_predictions': 'True',
            'feature_prefix': '1gram, 2gram',
            'subgroups': 'native language, GPA_range',
            'exclude_zero_scores': 'Yes'}

    process_json_fields(validate_and_populate_json_fields(data))

@raises(ValueError)
def test_process_fields_with_integer():
    data = {'experiment_id': 'experiment_1',
            'train_file': 'data/rsmtool_smTrain.csv',
            'test_file': 'data/rsmtool_smEval.csv',
            'description': 'Test',
            'model': 'empWt',
            'use_scaled_predictions': 'True',
            'feature_prefix': '1gram, 2gram',
            'subgroups': 'native language, GPA_range',
            'exclude_zero_scores': 1}

    process_json_fields(validate_and_populate_json_fields(data))


def test_rename_no_columns():
    df = pd.DataFrame(columns=['spkitemid', 'sc1', 'sc2', 'length', 'raw', 'candidate', 'feature1', 'feature2'])
    df = rename_default_columns(df, [], 'spkitemid', 'sc1', 'sc2', 'length', 'raw', 'candidate')
    assert_array_equal(df.columns, ['spkitemid', 'sc1', 'sc2', 'length', 'raw', 'candidate', 'feature1', 'feature2'])


def test_rename_no_columns_some_values_none():
    df = pd.DataFrame(columns=['spkitemid', 'sc1', 'sc2', 'feature1', 'feature2'])
    df = rename_default_columns(df, [], 'spkitemid', 'sc1', 'sc2', None, None, None)
    assert_array_equal(df.columns, ['spkitemid', 'sc1', 'sc2', 'feature1', 'feature2'])


def test_rename_no_used_columns_but_unused_columns_with_default_names():
    df = pd.DataFrame(columns=['spkitemid', 'sc1', 'sc2', 'length', 'feature1', 'feature2'])
    df = rename_default_columns(df, [], 'spkitemid', 'sc1', 'sc2', None, None, None)
    assert_array_equal(df.columns, ['spkitemid', 'sc1', 'sc2', '##length##', 'feature1', 'feature2'])


def test_rename_used_columns():
    df = pd.DataFrame(columns=['id', 'r1', 'r2', 'words', 'SR', 'feature1', 'feature2'])
    df = rename_default_columns(df, [], 'id', 'r1', 'r2', 'words', 'SR', None)
    assert_array_equal(df.columns, ['spkitemid', 'sc1', 'sc2', 'length', 'raw', 'feature1', 'feature2'])


def test_rename_used_columns_and_unused_columns_with_default_names():
    df = pd.DataFrame(columns=['id', 'r1', 'r2', 'words', 'raw', 'feature1', 'feature2'])
    df = rename_default_columns(df, [], 'id', 'r1', 'r2', 'words', None, None)
    assert_array_equal(df.columns, ['spkitemid', 'sc1', 'sc2', 'length', '##raw##', 'feature1', 'feature2'])


def test_rename_used_columns_with_swapped_names():
    df = pd.DataFrame(columns=['id', 'sc1', 'sc2', 'raw', 'words', 'feature1', 'feature2'])
    df = rename_default_columns(df, [], 'id', 'sc2', 'sc1', 'words', None, None)
    assert_array_equal(df.columns, ['spkitemid', 'sc2', 'sc1', '##raw##', 'length', 'feature1', 'feature2'])


def test_rename_used_columns_but_not_features():
    df = pd.DataFrame(columns=['id', 'sc1', 'sc2', 'length', 'feature2'])
    df = rename_default_columns(df, ['length'], 'id', 'sc1', 'sc2', None, None, None)
    assert_array_equal(df.columns, ['spkitemid', 'sc1', 'sc2', 'length', 'feature2'])


def test_rename_candidate_column():
    df = pd.DataFrame(columns=['spkitemid', 'sc1', 'sc2', 'length', 'apptNo', 'feature1', 'feature2'])
    df = rename_default_columns(df, [], 'spkitemid', 'sc1', 'sc2', None, None, 'apptNo')
    assert_array_equal(df.columns, ['spkitemid', 'sc1', 'sc2', '##length##', 'candidate', 'feature1', 'feature2'])


def test_rename_candidate_named_sc2():
    df = pd.DataFrame(columns=['id', 'sc1', 'sc2', 'question', 'l1', 'score'])
    df_renamed = rename_default_columns(df, [], 'id', 'sc1', None, None, 'score', 'sc2')
    assert_array_equal(df_renamed.columns, ['spkitemid', 'sc1', 'candidate', 'question', 'l1', 'raw'])


def test_check_flag_column():
    input_dict = {"advisory flag": ['0']}
    config = {"flag_column": input_dict}
    output_dict = check_flag_column(config)
    eq_(input_dict, output_dict)


def test_check_flag_column_keep_numeric():
    input_dict = {"advisory flag": [1, 2, 3]}
    config = {"flag_column": input_dict}
    output_dict = check_flag_column(config)
    eq_(output_dict, {"advisory flag": [1, 2, 3]})


def test_check_flag_column_no_values():
    config = {"flag_column": None}
    flag_dict = check_flag_column(config)
    eq_(flag_dict, {})


def test_check_flag_column_convert_to_list():
    config = {"flag_column": {"advisories": "0"}}
    flag_dict = check_flag_column(config)
    eq_(flag_dict, {"advisories": ['0']})


def test_check_flag_column_convert_to_list_keep_numeric():
    config = {"flag_column": {"advisories": 123}}
    flag_dict = check_flag_column(config)
    eq_(flag_dict, {"advisories": [123]})


@raises(ValueError)
def test_check_flag_column_wrong_format():
    config = {"flag_column": "[advisories]"}
    check_flag_column(config)

@raises(KeyError)
def test_check_subgroups_missing_columns():
    df = pd.DataFrame(columns=['a', 'b', 'c'])
    subgroups = ['a', 'd']
    check_subgroups(df, subgroups)


def test_check_subgroups_nothing_to_replace():
    df = pd.DataFrame({'a': ['1', '2'],
                       'b': ['32', '34'],
                       'd': ['abc', 'def']})
    subgroups = ['a', 'd']
    df_out = check_subgroups(df, subgroups)
    assert_frame_equal(df_out, df)


def test_check_subgroups_replace_empty():
    df = pd.DataFrame({'a': ['1', ''],
                       'b': ['   ', '34'],
                       'd': ['ab c', '   ']})
    subgroups = ['a', 'd']
    df_expected = pd.DataFrame({'a': ['1', 'No info'],
                               'b': ['   ', '34'],
                               'd': ['ab c', 'No info']})
    df_out = check_subgroups(df, subgroups)
    assert_frame_equal(df_out, df_expected)


@raises(ValueError)
def test_check_feature_subset_file_no_feature_column():
    feature_specs = pd.DataFrame({'feat': ['f1', 'f2', 'f3'], 'subset1': [0, 1, 0]})
    check_feature_subset_file(feature_specs, 'subset1')


@raises(ValueError)
def test_check_feature_subset_file_no_subset_column():
    feature_specs = pd.DataFrame({'Feature': ['f1', 'f2', 'f3'], 'subset1': [0, 1, 0]})
    check_feature_subset_file(feature_specs, 'subset2')


@raises(ValueError)
def test_check_feature_subset_file_wrong_values_in_subset():
    feature_specs = pd.DataFrame({'Feature': ['f1', 'f2', 'f3'], 'subset1': ['yes', 'no', 'yes']})
    check_feature_subset_file(feature_specs, 'subset1')

@raises(ValueError)
def test_check_feature_subset_file_no_sign_column():
    feature_specs = pd.DataFrame({'feature': ['f1', 'f2', 'f3'],
                                  'subset1': [0, 1, 0]})
    check_feature_subset_file(feature_specs, sign='subset1')


@raises(ValueError)
def test_check_feature_subset_file_wrong_values_in_sign():
    feature_specs = pd.DataFrame({'Feature': ['f1', 'f2', 'f3'],
                                  'sign_SYS1': ['+1', '-1', '+1']})
    check_feature_subset_file(feature_specs, sign='SYS1')


def test_select_candidates_with_N_or_more_items():
    data = pd.DataFrame({'candidate': ['a']*3 + ['b']*2 + ['c'],
                         'sc1' : [2, 3, 1, 5, 6, 1]})
    df_included_expected = pd.DataFrame({'candidate': ['a']*3 + ['b']*2,
                                         'sc1' : [2, 3, 1, 5, 6]})
    df_excluded_expected = pd.DataFrame({'candidate': ['c'],
                                         'sc1' : [1]})
    (df_included,
     df_excluded) = select_candidates_with_N_or_more_items(data, 2)
    assert_frame_equal(df_included, df_included_expected)
    assert_frame_equal(df_excluded, df_excluded_expected)


def test_select_candidates_with_N_or_more_items_all_included():
    data = pd.DataFrame({'candidate': ['a']*2 + ['b']*2 + ['c']*2,
                         'sc1' : [2, 3, 1, 5, 6, 1]})
    (df_included,
     df_excluded) = select_candidates_with_N_or_more_items(data, 2)
    assert_frame_equal(df_included, data)
    assert_equal(len(df_excluded), 0)


def test_select_candidates_with_N_or_more_items_all_excluded():
    data = pd.DataFrame({'candidate': ['a']*3 + ['b']*2 + ['c'],
                         'sc1' : [2, 3, 1, 5, 6, 1]})
    (df_included,
     df_excluded) = select_candidates_with_N_or_more_items(data, 4)
    assert_frame_equal(df_excluded, data)
    assert_equal(len(df_included), 0)

def test_select_candidates_with_N_or_more_items_custom_name():
    data = pd.DataFrame({'ID': ['a']*3 + ['b']*2 + ['c'],
                         'sc1' : [2, 3, 1, 5, 6, 1]})
    df_included_expected = pd.DataFrame({'ID': ['a']*3 + ['b']*2,
                                         'sc1' : [2, 3, 1, 5, 6]})
    df_excluded_expected = pd.DataFrame({'ID': ['c'],
                                         'sc1' : [1]})
    (df_included,
     df_excluded) = select_candidates_with_N_or_more_items(data, 2, 'ID')
    assert_frame_equal(df_included, df_included_expected)
    assert_frame_equal(df_excluded, df_excluded_expected)


def test_normalize_and_validate_json_feature_file():
    feature_json = {'features': [{'feature': 'f1',
                                  'transform': 'raw',
                                  'sign': 1},
                                 {'feature': 'f2',
                                  'transform': 'inv',
                                  'sign' : -1}]}
    new_feature_json = normalize_and_validate_json_feature_file(feature_json)
    assert_equal(new_feature_json, feature_json)


def test_normalize_json_feature_file_old_file():
    old_feature_json = {'feats': [{'featN': 'f1',
                                  'trans': 'raw',
                                  'wt': 1},
                                 {'featN': 'f2',
                                  'trans': 'inv',
                                  'wt' : -1}]}
    expected_feature_json = {'features': [{'feature': 'f1',
                                  'transform': 'raw',
                                  'sign': 1},
                                 {'feature': 'f2',
                                  'transform': 'inv',
                                  'sign' : -1}]}
    new_feature_json = normalize_and_validate_json_feature_file(old_feature_json)
    assert_equal(new_feature_json, expected_feature_json)


@raises(KeyError)
def test_normalize_and_validate_json_feature_file_missing_fields():
    feature_json = {'features': [{'feature': 'f1',
                                  'sign': 1},
                                 {'feature': 'f2',
                                  'transform': 'inv'}]}
    normalize_and_validate_json_feature_file(feature_json)


def test_validate_feature_specs():
    df_feature_specs = pd.DataFrame({'feature': ['f1', 'f2', 'f3'],
                                     'transform': ['raw', 'inv', 'sqrt'],
                                     'sign': [1.0, 1.0, -1.0]})
    df_new_feature_specs = validate_feature_specs(df_feature_specs)
    assert_frame_equal(df_feature_specs, df_new_feature_specs)


def test_validate_feature_specs_with_Feature_as_column():
    df_feature_specs = pd.DataFrame({'Feature': ['f1', 'f2', 'f3'],
                                     'transform': ['raw', 'inv', 'sqrt'],
                                     'sign': [1.0, 1.0, -1.0]})
    df_expected_feature_specs = pd.DataFrame({'feature': ['f1', 'f2', 'f3'],
                                         'transform': ['raw', 'inv', 'sqrt'],
                                         'sign': [1.0, 1.0, -1.0]})
    df_new_feature_specs = validate_feature_specs(df_feature_specs)
    assert_frame_equal(df_new_feature_specs, df_expected_feature_specs)



def test_validate_feature_specs_sign_to_float():
    df_feature_specs = pd.DataFrame({'feature': ['f1', 'f2', 'f3'],
                                     'transform': ['raw', 'inv', 'sqrt'],
                                     'sign': ['1', '1', '-1']})
    df_expected_feature_specs = pd.DataFrame({'feature': ['f1', 'f2', 'f3'],
                                              'transform': ['raw', 'inv', 'sqrt'],
                                              'sign': [1.0, 1.0, -1.0]})
    df_new_feature_specs = validate_feature_specs(df_feature_specs)
    assert_frame_equal(df_new_feature_specs, df_expected_feature_specs)


def test_validate_feature_specs_add_default_values():
    df_feature_specs = pd.DataFrame({'feature': ['f1', 'f2', 'f3']})
    df_expected_feature_specs = pd.DataFrame({'feature': ['f1', 'f2', 'f3'],
                                              'transform': ['raw', 'raw', 'raw'],
                                              'sign': [1, 1, 1]})
    df_new_feature_specs = validate_feature_specs(df_feature_specs)
    assert_frame_equal(df_new_feature_specs, df_expected_feature_specs)


@raises(ValueError)
def test_validate_feature_specs_wrong_sign_format():
    df_feature_specs = pd.DataFrame({'feature': ['f1', 'f2', 'f3'],
                                     'transform': ['raw', 'inv', 'sqrt'],
                                     'sign': ['+', '+', '-']})
    validate_feature_specs(df_feature_specs)


@raises(ValueError)
def test_validate_feature_duplicate_feature():
    df_feature_specs = pd.DataFrame({'feature': ['f1', 'f1', 'f3'],
                                     'transform': ['raw', 'inv', 'sqrt'],
                                     'sign': ['+', '+', '-']})
    validate_feature_specs(df_feature_specs)


@raises(KeyError)
def test_validate_feature_missing_feature_column():
    df_feature_specs = pd.DataFrame({'FeatureName': ['f1', 'f1', 'f3'],
                                     'transform': ['raw', 'inv', 'sqrt'],
                                     'sign': ['+', '+', '-']})
    validate_feature_specs(df_feature_specs)


def test_json_feature_conversion():
    json_feature_file = join(_MY_DIR, 'data', 'experiments', 'lr-feature-json', 'features.json')
    expected_feature_csv = join(_MY_DIR, 'data', 'experiments', 'lr', 'features.csv')

    # convert the feature json file and write to a temporary location
    tempf = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    convert_feature_json_file(json_feature_file, tempf.name, delete=False)

    # read the expected and converted files into data frames
    df_expected = pd.read_csv(expected_feature_csv)
    df_converted = pd.read_csv(tempf.name)
    tempf.close()

    # get rid of the file now that have read it into memory
    os.unlink(tempf.name)

    assert_frame_equal(df_expected, df_converted)

@raises(RuntimeError)
def test_json_feature_conversion_bad_json():
    json_feature_file = join(_MY_DIR, 'data', 'experiments', 'lr-feature-json', 'lr.json')

    # convert the feature json file and write to a temporary location
    tempf = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
    convert_feature_json_file(json_feature_file, tempf.name, delete=False)

@raises(RuntimeError)
def test_json_feature_conversion_bad_output_file():
    json_feature_file = join(_MY_DIR, 'data', 'experiments', 'lr-feature-json', 'features.json')

    # convert the feature json file and write to a temporary location
    tempf = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    convert_feature_json_file(json_feature_file, tempf.name, delete=False)

