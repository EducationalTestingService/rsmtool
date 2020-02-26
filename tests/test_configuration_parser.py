import json
import logging
import os
import tempfile
import warnings
import pandas as pd

from io import StringIO
from os import getcwd
from os.path import abspath, dirname, join

from pathlib import Path

from shutil import rmtree

from numpy.testing import assert_array_equal
from pandas.util.testing import assert_frame_equal

from nose.tools import (assert_equal,
                        assert_not_equal,
                        assert_raises,
                        eq_,
                        ok_,
                        raises)

from rsmtool.convert_feature_json import convert_feature_json_file

from rsmtool.configuration_parser import (Configuration,
                                          ConfigurationParser,
                                          CFGConfigurationParser,
                                          JSONConfigurationParser)


_MY_DIR = dirname(__file__)


class TestConfigurationParser:

    def setUp(self):
        self.parser = ConfigurationParser()

    def test_normalize_config(self):
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

            # Add data to `ConfigurationParser` object
            self.parser.load_config_from_dict(data)

            newdata = self.parser.normalize_config()
            ok_('experiment_id' in newdata.keys())
            assert_equal(newdata['experiment_id'], 'experiment_1')
            assert_equal(newdata['use_scaled_predictions'], True)

        # test for non-standard scaling value
        data = {'expID': 'experiment_1',
                'train': 'data/rsmtool_smTrain.csv',
                'LRmodel': 'LinearRegression',
                'scale': 'Yes'}
        with warnings.catch_warnings():

            # Add data to `ConfigurationParser` object
            self.parser._config = data

            warnings.filterwarnings('ignore', category=DeprecationWarning)
            assert_raises(ValueError, self.parser.normalize_config)

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

            # Add data to `ConfigurationParser` object
            self.parser._config = data
            newdata = self.parser.normalize_config()
            ok_('use_scaled_predictions' not in newdata.keys())

    def test_load_config_from_dict(self):
        data = {'expID': 'test'}
        configdir = abspath('/path/to/configdir')
        self.parser.load_config_from_dict(data,
                                          configdir=configdir)
        eq_(self.parser._configdir, configdir)


    def test_load_config_from_dict_no_configdir(self):
        data = {'expID': 'test'}
        configdir = getcwd()
        self.parser.load_config_from_dict(data)
        eq_(self.parser._configdir, configdir)


    @raises(AttributeError)
    def test_load_config_from_dict_dict_already_assigned(self):
        data = {'expID': 'test'}
        configdir = 'some/config/dir'
        self.parser._config = data
        self.parser.load_config_from_dict(data, configdir)

    @raises(TypeError)
    def test_load_config_from_dict_wrong_type(self):
        data = [('expID', 'test')]
        configdir = 'some/config/dir'
        self.parser.load_config_from_dict(data, configdir)

    def test_load_and_normalize_config_from_dict_rsmtool(self):
        data = {'experiment_id': 'experiment_1',
                'train_file': 'data/rsmtool_smTrain.csv',
                'test_file': 'data/rsmtool_smEval.csv',
                'model': 'LinearRegression'}

        # Add data to `ConfigurationParser` object
        newdata = self.parser.load_normalize_and_validate_config_from_dict(data)
        assert_equal(newdata['id_column'], 'spkitemid')
        assert_equal(newdata['use_scaled_predictions'], False)
        assert_equal(newdata['select_transformations'], False)
        assert_array_equal(newdata['general_sections'], ['all'])
        assert_equal(newdata['description'], '')
        assert_equal(newdata.configdir, getcwd())


    def test_load_and_normalize_config_from_dict_rsmeval(self):
        data = {'experiment_id': 'experiment_1',
                'predictions_file': 'data/rsmtool_smTrain.csv',
                'system_score_column': 'system',
                'trim_min': 1,
                'trim_max': 5}

        # Add data to `ConfigurationParser` object
        newdata = self.parser.load_normalize_and_validate_config_from_dict(data,
                                                                         context='rsmeval')
        assert_equal(newdata['id_column'], 'spkitemid')
        assert_array_equal(newdata['general_sections'], ['all'])
        assert_equal(newdata['description'], '')
        assert_equal(newdata.configdir, getcwd())


    @raises(ValueError)
    def test_validate_config_missing_fields(self):
        data = {'expID': 'test'}

        # Add data to `ConfigurationParser` object
        self.parser.load_config_from_dict(data)
        self.parser.validate_config()

    @raises(ValueError)
    def test_validate_config_min_responses_but_no_candidate(self):
        data = {'experiment_id': 'experiment_1',
                'train_file': 'data/rsmtool_smTrain.csv',
                'test_file': 'data/rsmtool_smEval.csv',
                'model': 'LinearRegression',
                'min_responses_per_candidate': 5}

        # Add data to `ConfigurationParser` object
        self.parser.load_config_from_dict(data)
        self.parser.validate_config()

    def test_validate_config_unspecified_fields(self):
        data = {'experiment_id': 'experiment_1',
                'train_file': 'data/rsmtool_smTrain.csv',
                'test_file': 'data/rsmtool_smEval.csv',
                'model': 'LinearRegression'}

        # Add data to `ConfigurationParser` object
        self.parser._config = data
        newdata = self.parser.validate_config()
        assert_equal(newdata['id_column'], 'spkitemid')
        assert_equal(newdata['use_scaled_predictions'], False)
        assert_equal(newdata['select_transformations'], False)
        assert_array_equal(newdata['general_sections'], ['all'])
        assert_equal(newdata['description'], '')

    @raises(ValueError)
    def test_validate_config_unknown_fields(self):
        data = {'experiment_id': 'experiment_1',
                'train_file': 'data/rsmtool_smTrain.csv',
                'test_file': 'data/rsmtool_smEval.csv',
                'description': 'Test',
                'model': 'LinearRegression',
                'output': 'foobar'}

        # Add data to `ConfigurationParser` object
        self.parser.load_config_from_dict(data)
        self.parser.validate_config()

    @raises(ValueError)
    def test_validate_config_experiment_id_1(self):
        data = {'experiment_id': 'test experiment',
                'train_file': 'data/rsmtool_smTrain.csv',
                'test_file': 'data/rsmtool_smEval.csv',
                'model': 'LinearRegression'}

        # Add data to `ConfigurationParser` object
        self.parser.load_config_from_dict(data)
        self.parser.validate_config()

    @raises(ValueError)
    def test_validate_config_experiment_id_2(self):
        data = {'experiment_id': 'test experiment',
                'predictions_file': 'data/foo',
                'system_score_column': 'h1',
                'trim_min': 1,
                'trim_max': 5}

        # Add data to `ConfigurationParser` object
        self.parser.load_config_from_dict(data)
        self.parser.validate_config(context='rsmeval')

    @raises(ValueError)
    def test_validate_config_experiment_id_3(self):
        data = {'comparison_id': 'old vs new',
                'experiment_id_old': 'old_experiment',
                'experiment_dir_old': 'data/old',
                'experiment_id_new': 'new_experiment',
                'experiment_dir_new': 'data/new'}

        # Add data to `ConfigurationParser` object
        self.parser.load_config_from_dict(data)
        self.parser.validate_config(context='rsmcompare')

    @raises(ValueError)
    def test_validate_config_experiment_id_4(self):
        data = {'comparison_id': 'old vs new',
                'experiment_id_old': 'old experiment',
                'experiment_dir_old': 'data/old',
                'experiment_id_new': 'new_experiment',
                'experiment_dir_new': 'data/new'}

        # Add data to `ConfigurationParser` object
        self.parser.load_config_from_dict(data)
        self.parser.validate_config(context='rsmcompare')

    @raises(ValueError)
    def test_validate_config_experiment_id_5(self):
        data = {'experiment_id': 'this_is_a_really_really_really_'
                'really_really_really_really_really_really_really_'
                'really_really_really_really_really_really_really_'
                'really_really_really_really_really_really_really_'
                'really_really_really_long_id',
                'train_file': 'data/rsmtool_smTrain.csv',
                'test_file': 'data/rsmtool_smEval.csv',
                'model': 'LinearRegression'}

        # Add data to `ConfigurationParser` object
        self.parser.load_config_from_dict(data)
        self.parser.validate_config()

    @raises(ValueError)
    def test_validate_config_experiment_id_6(self):
        data = {'experiment_id': 'this_is_a_really_really_really_'
                'really_really_really_really_really_really_really_'
                'really_really_really_really_really_really_really_'
                'really_really_really_really_really_really_really_'
                'really_really_really_long_id',
                'predictions_file': 'data/foo',
                'system_score_column': 'h1',
                'trim_min': 1,
                'trim_max': 5}

        # Add data to `ConfigurationParser` object
        self.parser.load_config_from_dict(data)
        self.parser.validate_config(context='rsmeval')

    @raises(ValueError)
    def test_validate_config_experiment_id_7(self):
        data = {'comparison_id': 'this_is_a_really_really_really_'
                'really_really_really_really_really_really_really_'
                'really_really_really_really_really_really_really_'
                'really_really_really_really_really_really_really_'
                'really_really_really_long_id',
                'experiment_id_old': 'old_experiment',
                'experiment_dir_old': 'data/old',
                'experiment_id_new': 'new_experiment',
                'experiment_dir_new': 'data/new'}

        # Add data to `ConfigurationParser` object
        self.parser.load_config_from_dict(data)
        self.parser.validate_config(context='rsmcompare')

    @raises(ValueError)
    def test_validate_config_experiment_id_8(self):
        data = {'summary_id': 'model summary',
                'experiment_dirs': []}

        # Add data to `ConfigurationParser` object
        self.parser.load_config_from_dict(data)
        self.parser.validate_config(context='rsmsummarize')

    @raises(ValueError)
    def test_validate_config_experiment_id_9(self):
        data = {'summary_id': 'this_is_a_really_really_really_'
                'really_really_really_really_really_really_really_'
                'really_really_really_really_really_really_really_'
                'really_really_really_really_really_really_really_'
                'really_really_really_long_id',
                'experiment_dirs': []}

        # Add data to `ConfigurationParser` object
        self.parser.load_config_from_dict(data)
        self.parser.validate_config(context='rsmsummarize')

    @raises(ValueError)
    def test_validate_config_too_many_experiment_names(self):
        data = {'summary_id': 'summary',
                'experiment_dirs': ["dir1", "dir2", "dir3"],
                'experiment_names': ['exp1', 'exp2', 'exp3', 'exp4']}

        # Add data to `ConfigurationParser` object
        self.parser.load_config_from_dict(data)
        self.parser.validate_config(context='rsmsummarize')

    @raises(ValueError)
    def test_validate_config_too_few_experiment_names(self):
        data = {'summary_id': 'summary',
                'experiment_dirs': ["dir1", "dir2", "dir3"],
                'experiment_names': ['exp1', 'exp2']}

        # Add data to `ConfigurationParser` object
        self.parser.load_config_from_dict(data)
        self.parser.validate_config(context='rsmsummarize')

    def test_validate_config_numeric_subgroup_threshold(self):
        data = {'experiment_id': 'experiment_1',
                'train_file': 'data/rsmtool_smTrain.csv',
                'test_file': 'data/rsmtool_smEval.csv',
                'model': 'LinearRegression',
                'subgroups': ['L2', 'L1'],
                'min_n_per_group': 100}
        self.parser._config = data
        newdata = self.parser.validate_config()
        eq_(type(newdata['min_n_per_group']), dict)
        assert_equal(newdata['min_n_per_group']['L1'], 100)
        assert_equal(newdata['min_n_per_group']['L2'], 100)

    def test_validate_config_dictionary_subgroup_threshold(self):
        data = {'experiment_id': 'experiment_1',
                'train_file': 'data/rsmtool_smTrain.csv',
                'test_file': 'data/rsmtool_smEval.csv',
                'model': 'LinearRegression',
                'subgroups': ['L2', 'L1'],
                'min_n_per_group': {"L1": 100,
                                    "L2": 200}}
        self.parser._config = data
        newdata = self.parser.validate_config()
        eq_(type(newdata['min_n_per_group']), dict)
        assert_equal(newdata['min_n_per_group']['L1'], 100)
        assert_equal(newdata['min_n_per_group']['L2'], 200)

    @raises(ValueError)
    def test_valdiate_config_too_few_subgroup_keys(self):
        data = {'experiment_id': 'experiment_1',
                'train_file': 'data/rsmtool_smTrain.csv',
                'test_file': 'data/rsmtool_smEval.csv',
                'model': 'LinearRegression',
                'subgroups': ['L1', 'L2'],
                'min_n_per_group': {"L1": 100}}
        self.parser.load_config_from_dict(data)
        self.parser.validate_config()

    @raises(ValueError)
    def test_valdiate_config_too_many_subgroup_keys(self):
        data = {'experiment_id': 'experiment_1',
                'train_file': 'data/rsmtool_smTrain.csv',
                'test_file': 'data/rsmtool_smEval.csv',
                'model': 'LinearRegression',
                'subgroups': ['L1', 'L2'],
                'min_n_per_group': {"L1": 100,
                                    "L2": 100,
                                    "L4": 50}}
        self.parser.load_config_from_dict(data)
        self.parser.validate_config()

    @raises(ValueError)
    def test_validate_config_mismatched_subgroup_keys(self):
        data = {'experiment_id': 'experiment_1',
                'train_file': 'data/rsmtool_smTrain.csv',
                'test_file': 'data/rsmtool_smEval.csv',
                'model': 'LinearRegression',
                'subgroups': ['L1', 'L2'],
                'min_n_per_group': {"L1": 100,
                                    "L4": 50}}
        self.parser.load_config_from_dict(data)
        self.parser.validate_config()

    @raises(ValueError)
    def test_validate_config_min_n_without_subgroups(self):
        data = {'experiment_id': 'experiment_1',
                'train_file': 'data/rsmtool_smTrain.csv',
                'test_file': 'data/rsmtool_smEval.csv',
                'model': 'LinearRegression',
                'min_n_per_group': {"L1": 100,
                                    "L2": 50}}
        self.parser.load_config_from_dict(data)
        self.parser.validate_config()

    def test_process_fields(self):
        data = {'experiment_id': 'experiment_1',
                'train_file': 'data/rsmtool_smTrain.csv',
                'test_file': 'data/rsmtool_smEval.csv',
                'description': 'Test',
                'model': 'empWt',
                'use_scaled_predictions': 'True',
                'feature_prefix': '1gram, 2gram',
                'subgroups': 'native language, GPA_range',
                'exclude_zero_scores': 'false'}

        # Add data to `ConfigurationParser` object
        self.parser.load_config_from_dict(data)
        newdata = self.parser.validate_config(inplace=False)

        # Add data to `ConfigurationParser` object
        self.parser._config = newdata
        newdata = self.parser.process_config(inplace=False)
        assert_array_equal(newdata['feature_prefix'], ['1gram', '2gram'])
        assert_array_equal(newdata['subgroups'], ['native language', 'GPA_range'])
        eq_(type(newdata['use_scaled_predictions']), bool)
        eq_(newdata['use_scaled_predictions'], True)
        eq_(newdata['exclude_zero_scores'], False)

    @raises(ValueError)
    def test_process_fields_with_non_boolean(self):
        data = {'experiment_id': 'experiment_1',
                'train_file': 'data/rsmtool_smTrain.csv',
                'test_file': 'data/rsmtool_smEval.csv',
                'description': 'Test',
                'model': 'empWt',
                'use_scaled_predictions': 'True',
                'feature_prefix': '1gram, 2gram',
                'subgroups': 'native language, GPA_range',
                'exclude_zero_scores': 'Yes'}

        # Add data to `ConfigurationParser` object
        self.parser.load_config_from_dict(data)
        newdata = self.parser.validate_config()
        # Add data to `ConfigurationParser` object
        self.parser._config = newdata
        newdata = self.parser.process_config()

    @raises(ValueError)
    def test_process_fields_with_integer(self):
        data = {'experiment_id': 'experiment_1',
                'train_file': 'data/rsmtool_smTrain.csv',
                'test_file': 'data/rsmtool_smEval.csv',
                'description': 'Test',
                'model': 'empWt',
                'use_scaled_predictions': 'True',
                'feature_prefix': '1gram, 2gram',
                'subgroups': 'native language, GPA_range',
                'exclude_zero_scores': 1}

        # Add data to `ConfigurationParser` object
        self.parser.load_config_from_dict(data)
        newdata = self.parser.validate_config()
        # Add data to `ConfigurationParser` object
        self.parser._config = newdata
        newdata = self.parser.process_config()

    def test_process_fields_rsmsummarize(self):
        data = {'summary_id': 'summary',
                'experiment_dirs': 'home/dir1, home/dir2, home/dir3',
                'experiment_names': 'exp1, exp2, exp3'}

        # Add data to `ConfigurationParser` object
        self.parser.load_config_from_dict(data)
        newdata = self.parser.process_config(inplace=False)

        assert_array_equal(newdata['experiment_dirs'], ['home/dir1',
                                                        'home/dir2',
                                                        'home/dir3'])
        assert_array_equal(newdata['experiment_names'], ['exp1',
                                                         'exp2',
                                                         'exp3'])

    @raises(ValueError)
    def test_invalid_skll_objective(self):
        data = {'experiment_id': 'experiment_1',
                'train_file': 'data/rsmtool_smTrain.csv',
                'test_file': 'data/rsmtool_smEval.csv',
                'description': 'Test',
                'model': 'LinearSVR',
                'skll_objective': 'squared_error'}

        # Add data to `ConfigurationParser` object
        self.parser.load_config_from_dict(data)
        self.parser.validate_config()

    @raises(ValueError)
    def test_wrong_skll_model_for_expected_scores(self):
        data = {'experiment_id': 'experiment_1',
                'train_file': 'data/rsmtool_smTrain.csv',
                'test_file': 'data/rsmtool_smEval.csv',
                'description': 'Test',
                'model': 'LinearSVR',
                'predict_expected_scores': 'true'}

        # Add data to `ConfigurationParser` object
        self.parser.load_config_from_dict(data)
        self.parser.validate_config()

    @raises(ValueError)
    def test_builtin_model_for_expected_scores(self):
        data = {'experiment_id': 'experiment_1',
                'train_file': 'data/rsmtool_smTrain.csv',
                'test_file': 'data/rsmtool_smEval.csv',
                'description': 'Test',
                'model': 'NNLR',
                'predict_expected_scores': 'true'}

        # Add data to `ConfigurationParser` object
        self.parser.load_config_from_dict(data)
        self.parser.validate_config()

    def test_proces_validate_correct_order_boolean(self):
        data = {'experiment_id': 'experiment_1',
                'train_file': 'data/rsmtool_smTrain.csv',
                'test_file': 'data/rsmtool_smEval.csv',
                'description': 'Test',
                'model': 'NNLR',
                'predict_expected_scores': 'false'}

        # Add data to `ConfigurationParser` object
        self.parser.load_config_from_dict(data)
        newdata = self.parser.normalize_validate_and_process_config()
        eq_(newdata['predict_expected_scores'], False)

    def test_process_validate_correct_order_list(self):
        data = {'summary_id': 'summary',
                'experiment_dirs': 'home/dir1, home/dir2, home/dir3',
                'experiment_names': 'exp1, exp2, exp3'}

        # Add data to `ConfigurationParser` object
        self.parser.load_config_from_dict(data)
        newdata = self.parser.normalize_validate_and_process_config(context='rsmsummarize')

        assert_array_equal(newdata['experiment_dirs'], ['home/dir1',
                                                        'home/dir2',
                                                        'home/dir3'])
        assert_array_equal(newdata['experiment_names'], ['exp1',
                                                         'exp2',
                                                         'exp3'])

    def test_get_correct_configparser_cfg(self):
        config_parser = ConfigurationParser.get_configparser('config.cfg')
        assert isinstance(config_parser, CFGConfigurationParser)

    def test_get_correct_configparser_json(self):
        config_parser = ConfigurationParser.get_configparser('config.json')
        assert isinstance(config_parser, JSONConfigurationParser)

    def test_get_correct_configparser_dict(self):
        config_parser = ConfigurationParser.get_configparser({'config': 'value'})
        assert isinstance(config_parser, ConfigurationParser)

    def test_get_correct_configparser_path(self):
        config_parser = ConfigurationParser.get_configparser(Path('config.json'))
        assert isinstance(config_parser, JSONConfigurationParser)

    @raises(ValueError)
    def test_get_correct_configparser_wrong_extension(self):
        config_parser = ConfigurationParser.get_configparser('config.txt')


class TestConfiguration:

    def test_init_default_values(self):
        config_dict = {'exp_id': 'my_experiment'}
        config = Configuration(config_dict)
        eq_(config._config, config_dict)
        eq_(config._filename, None)
        eq_(config.configdir, abspath(getcwd()))

    def test_init_with_filepath_as_positional_argument(self):
        with warnings.catch_warnings(record=True) as w:
            filepath = 'some/path/file.json'
            config_dict = {'exp_id': 'my_experiment'}
            config = Configuration(config_dict, filepath)
            eq_(config._filename, 'file.json')
            eq_(config._configdir, abspath('some/path'))
            assert len(w) == 1
            assert issubclass(w[-1].category, DeprecationWarning)

    def test_init_with_filename_as_kword_argument(self):
        filename = 'file.json'
        config_dict = {'exp_id': 'my_experiment'}
        config = Configuration(config_dict, filename=filename)
        eq_(config._filename, filename)
        eq_(config.configdir, abspath(getcwd()))

    def test_init_with_filename_and_configdir_as_kword_argument(self):
        filename = 'file.json'
        configdir = 'some/path'
        config_dict = {'exp_id': 'my_experiment'}
        config = Configuration(config_dict,
                               filename=filename,
                               configdir=configdir)
        eq_(config._filename, filename)
        eq_(config._configdir, abspath(configdir))


    def test_init_with_configdir_only_as_kword_argument(self):
        configdir = 'some/path'
        config_dict = {'exp_id': 'my_experiment'}
        config = Configuration(config_dict,
                               configdir=configdir)
        eq_(config._filename, None)
        eq_(config._configdir, abspath(configdir))

    @ raises(ValueError)
    def test_init_with_filepath_positional_and_filename_keyword(self):
        filepath = 'some/path/file.json'
        config_dict = {'exp_id': 'my_experiment'}
        _ = Configuration(config_dict,
                          filepath,
                          filename=filepath)

    @ raises(ValueError)
    def test_init_with_filepath_positional_and_configdir_keyword(self):
        filepath = 'some/path/file.json'
        configdir = 'some/path'
        config_dict = {'exp_id': 'my_experiment'}
        _ = Configuration(config_dict,
                          filepath,
                          configdir=configdir)


    def check_logging_output(self, expected, function, *args, **kwargs):

        # check if the `expected` text is in the actual logging output

        root_logger = logging.getLogger()
        with StringIO() as string_io:

            # add a stream handler
            handler = logging.StreamHandler(string_io)
            root_logger.addHandler(handler)

            result = function(*args, **kwargs)
            logging_text = string_io.getvalue()

            try:
                assert expected in logging_text
            except AssertionError:

                # remove the stream handler and raise error
                root_logger.handlers = []
                raise AssertionError('`{}` not in logging output: '
                                     '`{}`.'.format(expected, logging_text))

            # remove the stream handler, even if we have no errors
            root_logger.handlers = []
        return result

    def test_pop_value(self):
        dictionary = {"experiment_id": '001', 'trim_min': 1, 'trim_max': 6}
        config = Configuration(dictionary)
        value = config.pop("experiment_id")
        eq_(value, '001')

    def test_pop_value_default(self):
        dictionary = {"experiment_id": '001', 'trim_min': 1, 'trim_max': 6}
        config = Configuration(dictionary)
        value = config.pop("foo", "bar")
        eq_(value, 'bar')

    def test_copy(self):
        dictionary = {"experiment_id": '001', 'trim_min': 1, 'trim_max': 6,
                      "object": [1, 2, 3]}
        config = Configuration(dictionary)
        config_copy = config.copy()
        assert_not_equal(id(config), id(config_copy))
        for key in config.keys():

            # check to make sure this is a deep copy
            if key == "object":
                assert_not_equal(id(config[key]), id(config_copy[key]))
            assert_equal(config[key], config_copy[key])

    def test_copy_not_deep(self):
        dictionary = {"experiment_id": '001', 'trim_min': 1, 'trim_max': 6,
                      "object": [1, 2, 3]}
        config = Configuration(dictionary)
        config_copy = config.copy(deep=False)
        assert_not_equal(id(config), id(config_copy))
        for key in config.keys():

            # check to make sure this is a shallow copy
            if key == "object":
                assert_equal(id(config[key]), id(config_copy[key]))
            assert_equal(config[key], config_copy[key])

    def test_check_flag_column(self):
        input_dict = {"advisory flag": ['0']}
        config = Configuration({"flag_column": input_dict})
        output_dict = config.check_flag_column()
        eq_(input_dict, output_dict)

    def test_check_flag_column_flag_column_test(self):
        input_dict = {"advisory flag": ['0']}
        config = Configuration({"flag_column_test": input_dict})
        output_dict = config.check_flag_column("flag_column_test")
        eq_(input_dict, output_dict)

    def test_check_flag_column_keep_numeric(self):
        input_dict = {"advisory flag": [1, 2, 3]}
        config = Configuration({"flag_column": input_dict})
        output_dict = config.check_flag_column()
        eq_(output_dict, {"advisory flag": [1, 2, 3]})

    def test_check_flag_column_no_values(self):
        config = Configuration({"flag_column": None})
        flag_dict = config.check_flag_column()
        eq_(flag_dict, {})

    def test_check_flag_column_convert_to_list(self):
        config = Configuration({"flag_column": {"advisories": "0"}})
        flag_dict = config.check_flag_column()
        eq_(flag_dict, {"advisories": ['0']})

    def test_check_flag_column_convert_to_list_test(self):
        config = Configuration({"flag_column": {"advisories": "0"}})

        flag_dict = self.check_logging_output('evaluating',
                                              config.check_flag_column,
                                              partition='test')
        eq_(flag_dict, {"advisories": ['0']})

    def test_check_flag_column_convert_to_list_train(self):
        config = Configuration({"flag_column": {"advisories": "0"}})
        flag_dict = self.check_logging_output('training',
                                              config.check_flag_column,
                                              partition='train')
        eq_(flag_dict, {"advisories": ['0']})

    def test_check_flag_column_convert_to_list_both(self):
        config = Configuration({"flag_column": {"advisories": "0"}})
        flag_dict = self.check_logging_output('training and evaluating',
                                              config.check_flag_column,
                                              partition='both')
        eq_(flag_dict, {"advisories": ['0']})

    def test_check_flag_column_convert_to_list_unknown(self):
        config = Configuration({"flag_column": {"advisories": "0"}})
        flag_dict = self.check_logging_output('training and/or evaluating',
                                              config.check_flag_column,
                                              partition='unknown')
        eq_(flag_dict, {"advisories": ['0']})

    @raises(AssertionError)
    def test_check_flag_column_convert_to_list_test_error(self):
        config = Configuration({"flag_column": {"advisories": "0"}})
        self.check_logging_output('training',
                                  config.check_flag_column,
                                  partition='test')

    def test_check_flag_column_convert_to_list_keep_numeric(self):
        config = Configuration({"flag_column": {"advisories": 123}})
        flag_dict = config.check_flag_column()
        eq_(flag_dict, {"advisories": [123]})

    def test_contains_key(self):
        config = Configuration({"flag_column": {"advisories": 123}})
        ok_('flag_column' in config, msg="Test 'flag_column' in config.")

    def test_does_not_contain_nested_key(self):
        config = Configuration({"flag_column": {"advisories": 123}})
        eq_('advisories' in config, False)

    def test_get_item(self):
        expected_item = {"advisories": 123}
        config = Configuration({"flag_column": expected_item})
        item = config['flag_column']
        eq_(item, expected_item)

    def test_set_item(self):
        expected_item = ["45", 34]
        config = Configuration({"flag_column": {"advisories": 123}})
        config['other_column'] = expected_item
        eq_(config['other_column'], expected_item)

    def test_check_len(self):
        expected_len = 2
        config = Configuration({"flag_column": {"advisories": 123}, 'other_column': 5})
        eq_(len(config), expected_len)

    @raises(ValueError)
    def test_check_flag_column_wrong_format(self):
        config = Configuration({"flag_column": "[advisories]"})
        config.check_flag_column()

    @raises(ValueError)
    def test_check_flag_column_wrong_partition(self):
        config = Configuration({"flag_column_test": {"advisories": 123}})
        config.check_flag_column(partition='eval')

    @raises(ValueError)
    def test_check_flag_column_mismatched_partition(self):
        config = Configuration({"flag_column_test": {"advisories": 123}})
        config.check_flag_column(flag_column='flag_column_test',
                                 partition='train')

    @raises(ValueError)
    def test_check_flag_column_mismatched_partition_both(self):
        config = Configuration({"flag_column_test": {"advisories": 123}})
        config.check_flag_column(flag_column='flag_column_test',
                                 partition='both')

    def test_str_correct(self):
        config_dict = {'flag_column': '[advisories]'}
        config = Configuration(config_dict)
        print(config)
        eq_(config.__str__(), 'flag_column')

    # this is a test for an attribute that will be
    # deprecated
    def test_get_filepath(self):
        with warnings.catch_warnings(record=True) as warning_list:
            filepath = '/path/to/file.json'
            config = Configuration({"flag_column": "[advisories]"},
                                   configdir='/path/to/',
                                   filename='file.json')
            eq_(config.filepath, abspath(filepath))
            assert len(warning_list) == 1  # we get one deprecation warning here
            assert issubclass(warning_list[0].category, DeprecationWarning)


    # this is a test for an attribute that will be
    # deprecated
    def test_set_filepath(self):
        with warnings.catch_warnings(record=True) as warning_list:
            new_file_path = '/newpath/to/new.json'
            config = Configuration({"flag_column": "[advisories]"},
                                   configdir='/path/to/',
                                   filename='file.json',)
            config.filepath = new_file_path  # first deprecation warning
            eq_(config._filename, 'new.json')
            eq_(config.configdir, abspath('/newpath/to'))
            eq_(config.filepath, abspath(new_file_path))  # second deprecation warning
            assert len(warning_list) == 2  # we get two deprecation warnings here
            assert issubclass(warning_list[0].category, DeprecationWarning)
            assert issubclass(warning_list[1].category, DeprecationWarning)

    def test_get_configdir(self):
        configdir = '/path/to/dir/'
        config = Configuration({"flag_column": "[advisories]"},
                               configdir=configdir)
        eq_(config.configdir, abspath(configdir))

    def test_set_configdir(self):
        configdir = '/path/to/dir/'
        new_configdir = 'path/that/is/new/'
        config = Configuration({"flag_column": "[advisories]"},
                               configdir=configdir)
        config.configdir = new_configdir
        eq_(config.configdir, abspath(new_configdir))

    @raises(ValueError)
    def test_set_configdir_to_none(self):
        configdir = '/path/to/dir/'
        config = Configuration({"flag_column": "[advisories]"},
                               configdir=configdir)
        config.configdir = None


    def test_get_filename(self):
        filename = 'file.json'
        config = Configuration({"flag_column": "[advisories]"},
                               filename=filename)
        eq_(config.filename, filename)

    def test_set_filename(self):
        filename = 'file.json'
        new_filename = 'new_file.json'
        config = Configuration({"flag_column": "[advisories]"},
                               filename=filename)
        config.filename = new_filename
        eq_(config.filename, new_filename)

    def test_get_context(self):
        context = 'rsmtool'
        config = Configuration({"flag_column": "[advisories]"},
                               context=context)
        eq_(config.context, context)

    def test_set_context(self):
        context = 'rsmtool'
        new_context = 'rsmcompare'
        config = Configuration({"flag_column": "[advisories]"},
                               context=context)
        config.context = new_context
        eq_(config.context, new_context)

    def test_get(self):
        config = Configuration({"flag_column": "[advisories]"})
        eq_(config.get('flag_column'), "[advisories]")
        eq_(config.get('fasdfasfasdfa', 'hi'), 'hi')

    def test_to_dict(self):
        dictionary = {"flag_column": "abc", "other_column": 'xyz'}
        config = Configuration(dictionary)
        eq_(config.to_dict(), dictionary)

    def test_keys(self):
        dictionary = {"flag_column": "abc", "other_column": 'xyz'}
        keys = ['flag_column', 'other_column']
        config = Configuration(dictionary)
        eq_(sorted(config.keys()), sorted(keys))

    def test_values(self):
        dictionary = {"flag_column": "abc", "other_column": 'xyz'}
        values = ['abc', 'xyz']
        config = Configuration(dictionary)
        eq_(sorted(config.values()), sorted(values))

    def test_items(self):
        dictionary = {"flag_column": "abc", "other_column": 'xyz'}
        items = [('flag_column', 'abc'), ('other_column', 'xyz')]
        config = Configuration(dictionary)
        eq_(sorted(config.items()), sorted(items))

    def test_save(self):
        dictionary = {"experiment_id": '001', "flag_column": "abc"}
        config = Configuration(dictionary)
        config.save()

        out_path = 'output/001_rsmtool.json'
        with open(out_path) as buff:
            config_new = json.loads(buff.read())
        rmtree('output')
        eq_(config_new, dictionary)

    def test_save_rsmcompare(self):
        dictionary = {"comparison_id": '001'}
        config = Configuration(dictionary,
                               context='rsmcompare')
        config.save()

        out_path = 'output/001_rsmcompare.json'
        with open(out_path) as buff:
            config_new = json.loads(buff.read())
        rmtree('output')
        eq_(config_new, dictionary)

    def test_save_rsmsummarize(self):
        dictionary = {"summary_id": '001'}
        config = Configuration(dictionary,
                               context='rsmsummarize')
        config.save()

        out_path = 'output/001_rsmsummarize.json'
        with open(out_path) as buff:
            config_new = json.loads(buff.read())
        rmtree('output')
        eq_(config_new, dictionary)

    def test_check_exclude_listwise_true(self):
        dictionary = {"experiment_id": '001', "min_items_per_candidate": 4}
        config = Configuration(dictionary)
        exclude_list_wise = config.check_exclude_listwise()
        eq_(exclude_list_wise, True)

    def test_check_exclude_listwise_false(self):
        dictionary = {"experiment_id": '001'}
        config = Configuration(dictionary)
        exclude_list_wise = config.check_exclude_listwise()
        eq_(exclude_list_wise, False)

    def test_get_trim_min_max_tolerance_none(self):
        dictionary = {"experiment_id": '001'}
        config = Configuration(dictionary)
        trim_min_max_tolerance = config.get_trim_min_max_tolerance()
        eq_(trim_min_max_tolerance, (None, None, None))

    def test_get_trim_min_max_no_tolerance(self):
        dictionary = {"experiment_id": '001', 'trim_min': 1, 'trim_max': 6}
        config = Configuration(dictionary)
        trim_min_max_tolerance = config.get_trim_min_max_tolerance()
        eq_(trim_min_max_tolerance, (1.0, 6.0, None))

    def test_get_trim_min_max_values_tolerance(self):
        dictionary = {"experiment_id": '001',
                      'trim_min': 1,
                      'trim_max': 6,
                      'trim_tolerance': 0.49}
        config = Configuration(dictionary)
        trim_min_max_tolerance = config.get_trim_min_max_tolerance()
        eq_(trim_min_max_tolerance, (1.0, 6.0, 0.49))

    def test_get_trim_min_max_deprecated(self):
        dictionary = {"experiment_id": '001', 'trim_min': 1, 'trim_max': 6}
        config = Configuration(dictionary)
        trim_min_max = config.get_trim_min_max()
        eq_(trim_min_max, (1.0, 6.0))

    def test_get_trim_tolerance_no_min_max(self):
        dictionary = {"experiment_id": '001',
                      'trim_tolerance': 0.49}
        config = Configuration(dictionary)
        trim_min_max_tolerance = config.get_trim_min_max_tolerance()
        eq_(trim_min_max_tolerance, (None, None, 0.49))

    def test_get_names_and_paths_with_feature_file(self):
        filepaths = ['path/to/train.tsv',
                     'path/to/test.tsv',
                     'path/to/features.csv']
        filenames = ['train', 'test', 'feature_specs']

        expected = (filenames, filepaths)

        dictionary = {'id_column': 'A',
                      'candidate_column': 'B',
                      'train_file': 'path/to/train.tsv',
                      'test_file': 'path/to/test.tsv',
                      'features': 'path/to/features.csv',
                      'subgroups': ['C']}
        config = Configuration(dictionary)
        values_for_reader = config.get_names_and_paths(['train_file', 'test_file',
                                                        'features'],
                                                       ['train', 'test',
                                                        'feature_specs'])
        eq_(values_for_reader, expected)

    def test_get_names_and_paths_with_feature_subset(self):

        filepaths = ['path/to/train.tsv',
                     'path/to/test.tsv',
                     'path/to/feature_subset.csv']
        filenames = ['train', 'test', 'feature_subset_specs']

        expected = (filenames, filepaths)

        dictionary = {'id_column': 'A',
                      'candidate_column': 'B',
                      'train_file': 'path/to/train.tsv',
                      'test_file': 'path/to/test.tsv',
                      'feature_subset_file': 'path/to/feature_subset.csv',
                      'subgroups': ['C']}
        config = Configuration(dictionary)
        values_for_reader = config.get_names_and_paths(['train_file', 'test_file',
                                                        'feature_subset_file'],
                                                       ['train', 'test',
                                                        'feature_subset_specs'])
        eq_(values_for_reader, expected)

    def test_get_names_and_paths_with_feature_list(self):

        filepaths = ['path/to/train.tsv',
                     'path/to/test.tsv']
        filenames = ['train', 'test']

        expected = (filenames, filepaths)

        dictionary = {'id_column': 'A',
                      'candidate_column': 'B',
                      'train_file': 'path/to/train.tsv',
                      'test_file': 'path/to/test.tsv',
                      'features': ['FEATURE1', 'FEATURE2'],
                      'subgroups': ['C']}
        config = Configuration(dictionary)
        values_for_reader = config.get_names_and_paths(['train_file', 'test_file',
                                                        'features'],
                                                       ['train', 'test',
                                                        'feature_specs'])
        eq_(values_for_reader, expected)


class TestJSONFeatureConversion:

    def test_json_feature_conversion(self):
        json_feature_file = join(_MY_DIR, 'data', 'experiments',
                                 'lr-feature-json', 'features.json')
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
    def test_json_feature_conversion_bad_json(self):
        json_feature_file = join(_MY_DIR, 'data', 'experiments', 'lr-feature-json', 'lr.json')

        # convert the feature json file and write to a temporary location
        tempf = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
        convert_feature_json_file(json_feature_file, tempf.name, delete=False)

    @raises(RuntimeError)
    def test_json_feature_conversion_bad_output_file(self):
        json_feature_file = join(_MY_DIR, 'data', 'experiments',
                                 'lr-feature-json', 'features.json')

        # convert the feature json file and write to a temporary location
        tempf = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
        convert_feature_json_file(json_feature_file, tempf.name, delete=False)
