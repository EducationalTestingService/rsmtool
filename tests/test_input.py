import pandas as pd

from nose.tools import assert_equal, assert_raises, eq_, ok_, raises
from numpy.testing import assert_array_equal
from pandas.util.testing import assert_frame_equal

from rsmtool.input import (normalize_json_fields,
                           process_json_fields,
                           rename_default_columns,
                           validate_and_populate_json_fields,
                           check_flag_column,
                           check_subgroups,
                           check_feature_subset_file,
                           select_candidates_with_N_or_more_items)


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

    newdata = normalize_json_fields(data)
    ok_('experiment_id' in newdata.keys())
    assert_equal(newdata['experiment_id'], 'experiment_1')
    assert_equal(newdata['use_scaled_predictions'], True)

    # test for non-standard scaling value
    data = {'expID': 'experiment_1',
            'train': 'data/rsmtool_smTrain.csv',
            'LRmodel': 'LinearRegression',
            'scale': 'Yes'}
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
    data = {'experiment_id': 'test experiment',
            'experiment_dir': 'data/foobar',
            'input_features_file': 'data/feats'}
    validate_and_populate_json_fields(data, context='rsmpredict')


@raises(ValueError)
def test_validate_experiment_id_4():
    data = {'experiment_id_old': 'old experiment',
            'experiment_dir_old': 'data/old',
            'experiment_id_new': 'new experiment',
            'experiment_dir_new': 'data/new',}
    validate_and_populate_json_fields(data, context='rsmcompare')


@raises(ValueError)
def test_validate_experiment_id_5():
    data = {'experiment_id_old': 'old_experiment',
            'experiment_dir_old': 'data/old',
            'experiment_id_new': 'new experiment',
            'experiment_dir_new': 'data/new',}
    validate_and_populate_json_fields(data, context='rsmcompare')


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

    newdata = process_json_fields(validate_and_populate_json_fields(data))

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
            'exclude_zero_scores': '0'}

    newdata = process_json_fields(validate_and_populate_json_fields(data))


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
    feature_specs = pd.DataFrame({'feature': ['f1', 'f2', 'f3'], 'subset1': [0, 1, 0]})
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