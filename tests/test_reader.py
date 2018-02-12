import os
import tempfile

import pandas as pd

from nose.tools import raises, eq_
from pandas.util.testing import assert_frame_equal
from shutil import rmtree

from rsmtool.reader import DataReader


class TestDataReader:

    def setUp(self):
        self.filepaths = []

        # Create files
        self.df_train = pd.DataFrame({'id': ['001', '002', '003'],
                                      'feature1': [1, 2, 3],
                                      'feature2': [4, 5, 6],
                                      'gender': ['M', 'F', 'F'],
                                      'candidate': ['123', '456', '78901']})

        self.df_test = pd.DataFrame({'id': ['102', '102', '103'],
                                     'feature1': [5, 3, 2],
                                     'feature2': [3, 4, 3],
                                     'gender': ['F', 'M', 'F'],
                                     'candidate': ['135', '546', '781']})

        self.df_specs = pd.DataFrame({'feature': ['f1', 'f2', 'f3'],
                                      'transform': ['raw', 'inv', 'sqrt'],
                                      'sign': ['+', '+', '-']})

        self.df_other = pd.DataFrame({'random': ['a', 'b', 'c'],
                                      'things': [1241, 45332, 3252]})

    def tearDown(self):
        for path in self.filepaths:
            if os.path.exists(path):
                os.unlink(path)
        self.filepaths = []

    @staticmethod
    def make_file_from_ext(df, ext):
        tempf = tempfile.NamedTemporaryFile(mode='w',
                                            suffix='.{}'.format(ext),
                                            delete=False)
        if ext.lower() == 'csv':
            df.to_csv(tempf.name, index=False)
        elif ext.lower() == 'tsv':
            df.to_csv(tempf.name, sep='\t', index=False)
        elif ext.lower() in ['xls', 'xlsx']:
            df.to_excel(tempf.name, index=False)
        tempf.close()
        return tempf.name

    def get_container(self, name_ext_tuples, converters=None):
        """
        Get a DataContainer object from a list of tuples with (`name`, `ext`)
        """
        names_ = []
        paths_ = []
        for name, ext in name_ext_tuples:
            if name == 'train':
                df = self.df_train
            elif name == 'test':
                df = self.df_test
            elif name == 'feature_specs':
                df = self.df_specs
            else:
                df = self.df_other

            path = TestDataReader.make_file_from_ext(df, ext)

            names_.append(name)
            paths_.append(path)

        reader = DataReader(paths_, names_, converters)
        container = reader.read()

        self.filepaths.extend(paths_)
        return container

    def check_read_from_file(self, extension):
        """
        Test whether the ``read_from_file()`` method works as expected.
        """

        name = TestDataReader.make_file_from_ext(self.df_train, extension)

        # now read in the file using `read_data_file()`
        df_read = DataReader.read_from_file(name,
                                            converters={'id': str, 'candidate': str})

        # Make sure we get rid of the file at the end,
        # at least if we get to this point (i.e. no errors raised)
        self.filepaths.append(name)

        assert_frame_equal(self.df_train, df_read)

    def check_train(self, name_ext_tuples, converters=None):
        container = self.get_container(name_ext_tuples, converters)
        frame = container.train
        assert_frame_equal(frame, self.df_train)

    def check_feature_specs(self, name_ext_tuples, converters=None):
        container = self.get_container(name_ext_tuples, converters)
        frame = container.feature_specs
        assert_frame_equal(frame, self.df_specs)

    def test_read_data_file(self):
        # note that we cannot check for capital .xls and .xlsx
        # because xlwt does not support these extensions
        for extension in ['csv', 'tsv', 'xls', 'xlsx', 'CSV', 'TSV']:
            yield self.check_read_from_file, extension

    @raises(ValueError)
    def test_read_data_file_wrong_extension(self):
        self.check_read_from_file('txt')

    def test_container_train_property(self):
        test_lists = [[('train', 'csv'), ('test', 'tsv')],
                      [('train', 'csv'), ('feature_specs', 'xlsx')],
                      [('train', 'csv'), ('test', 'xls'), ('train_metadata', 'tsv')]]

        converter = {'id': str, 'feature1': int, 'feature2': int, 'candidate': str}
        converters = [{'train': converter, 'test': converter},
                      {'train': converter},
                      {'train': converter, 'test': converter}]
        for idx, test_list in enumerate(test_lists):
            yield self.check_train, test_list, converters[idx]

    def test_container_feature_specs_property(self):
        test_lists = [[('feature_specs', 'csv'), ('test', 'tsv')],
                      [('train', 'csv'), ('feature_specs', 'xlsx')],
                      [('train', 'csv'), ('feature_specs', 'xls'), ('train_metadata', 'tsv')]]
        for test_list in test_lists:
            yield self.check_feature_specs, test_list

    @raises(AttributeError)
    def test_no_container_feature_specs_property(self):
        test_lists = [('train', 'csv'), ('test', 'tsv'), ('train_metadata', 'xlsx')]
        container = self.get_container(test_lists)
        container.feature_specs

    @raises(AttributeError)
    def test_no_container_test_property(self):
        test_lists = [('feature_specs', 'csv'), ('train', 'tsv'), ('train_metadata', 'xlsx')]
        container = self.get_container(test_lists)
        container.test

    def test_container_test_property_frame_equal(self):
        test_lists = [('feature_specs', 'csv'), ('test', 'tsv'), ('train_metadata', 'xlsx')]
        converter = {'id': str, 'feature1': int, 'feature2': int, 'candidate': str}
        converters = {'test': converter}
        container = self.get_container(test_lists, converters)
        frame = container.test
        assert_frame_equal(frame, self.df_test)

    def test_get_values(self):
        test_lists = [('feature_specs', 'csv')]
        container = self.get_container(test_lists)
        frame = container['feature_specs']
        assert_frame_equal(container.values()[0], frame)

    def test_length(self):
        test_lists = [('feature_specs', 'csv')]
        container = self.get_container(test_lists)
        eq_(len(container), 1)

    def test_get_path_default(self):
        test_lists = [('feature_specs', 'csv')]
        container = self.get_container(test_lists)
        eq_(container.get_path('aaa'), None)

    def test_getitem_test_from_key(self):
        test_lists = [('feature_specs', 'csv'), ('test', 'tsv'), ('train', 'xlsx')]
        converter = {'id': str, 'feature1': int, 'feature2': int, 'candidate': str}
        converters = {'train': converter, 'test': converter}
        container = self.get_container(test_lists, converters)
        frame = container['test']
        assert_frame_equal(frame, self.df_test)

    def test_add_containers(self):
        test_list1 = [('feature_specs', 'csv'), ('train', 'xlsx')]
        container1 = self.get_container(test_list1)

        test_list2 = [('test', 'csv'), ('train_metadata', 'tsv')]
        container2 = self.get_container(test_list2)

        container3 = container1 + container2
        names = sorted(container3.keys())
        eq_(names, ['feature_specs', 'test', 'train', 'train_metadata'])

    @raises(KeyError)
    def test_add_containers_duplicate_keys(self):
        test_list1 = [('feature_specs', 'csv'), ('train', 'xlsx')]
        container1 = self.get_container(test_list1)

        test_list2 = [('test', 'csv'), ('train', 'tsv')]
        container2 = self.get_container(test_list2)
        container1 + container2

    def test_locate_files_list(self):

        paths = ['file1.csv', 'file2.xlsx']
        config_dir = 'output'
        result = DataReader.locate_files(paths, config_dir)
        assert isinstance(result, list)
        eq_(result, [None, None])

    def test_locate_files_str(self):

        paths = 'file1.csv'
        config_dir = 'output'
        result = DataReader.locate_files(paths, config_dir)
        eq_(result, None)

    def test_locate_files_works(self):

        config_dir = 'temp_output'
        os.makedirs(config_dir, exist_ok=True)

        paths = 'file1.csv'
        full_path = os.path.abspath(os.path.join(config_dir, paths))
        open(full_path, 'a').close()

        result = DataReader.locate_files(paths, config_dir)
        rmtree(config_dir)
        eq_(result, full_path)

    @raises(ValueError)
    def test_locate_files_wrong_type(self):

        paths = {'file1.csv', 'file2.xlsx'}
        config_dir = 'output'
        DataReader.locate_files(paths, config_dir)


    @raises(ValueError)
    def test_setup_none_in_path(self):
        paths = ['path1.csv', None, 'path2.csv']
        framenames = ['train', 'test', 'features']
        reader = DataReader(paths, framenames)


