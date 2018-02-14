#!/usr/bin/env python

"""
This script is designed to update the test files.
It takes the suffix of the output file and updates
the test data using the files generated under `test_outputs`.
Note that this will overwrite the original test data so make sure that
the outputs work as expected before running the script.

:author: Anastassia Loukina
:author: Nitin Madnani
:date: Feburary 2018
"""

import argparse
import re
import shutil


from ast import literal_eval as eval
from importlib.machinery import SourceFileLoader
from inspect import getmembers, getsourcelines, isfunction
from os.path import dirname, exists, join

_MY_PATH = dirname(__file__)


def update_test_output(source, experiment_id, filename, output_dir):

    # get the file we want to copy over
    output_file = '{}/{}/output/{}_{}'.format(output_dir,
                                              source,
                                              experiment_id,
                                              filename)

    # get the test file we want to overwrite
    test_file = '{}/data/experiments/{}/output/{}_{}'.format(_MY_PATH,
                                                             source,
                                                             experiment_id,
                                                             filename)

    # check if the specified output file already exists in test data
    if exists(output_file):
        print('copying {} to {}'.format(output_file, test_file))
        shutil.copy(output_file, test_file)
    else:
        print('{} does not exist'.format(output_file))


def main():
    # set up an argument parser
    parser = argparse.ArgumentParser(prog='update_test_files.py')
    parser.add_argument('filename', help="The test output filename to update")
    parser.add_argument('outputs_dir', help="The path to test_outputs")

    # parse given command line arguments
    args = parser.parse_args()

    # import the test_experiment.py test module using SourceFileLoader
    # so that it could also be used with tests in RSMExtra provided
    # they folllow the same structure.
    # see http://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
    # for Python 3.5 solution

    # we need to keep track of updated functions since the importing method below
    # might lead to the same functions being part of the same module
    for test_suffix in ['rsmtool_1', 'rsmtool_2', 'rsmtool_3', 'rsmtool_4',
                        'rsmeval', 'rsmpredict', 'rsmsummarize', 'rsmcompare']:
        test_module = SourceFileLoader('test_experiment', join(_MY_PATH, 'test_experiment_{}.py'.format(test_suffix))).load_module()

        # iterate over all the members and focus on only the experiment functions
        # and try to get the source and the experiment ID since that's what we
        # need to update the test files
        for member_name, member_object in getmembers(test_module):
            if isfunction(member_object) and member_name.startswith('test_run_experiment'):
                function = member_object

                # first we check if it's the parameterized function and if so
                # we can easily get the source and the experiment ID from
                # the parameter list
                if member_name.endswith('parameterized'):
                    for param in function.parameterized_input:
                        source, experiment_id = param.args
                        update_test_output(source,
                                           experiment_id,
                                           args.filename,
                                           args.outputs_dir)
                # if it's another function, then we actually inspect the source
                # to get the source and experiment_id
                else:
                    function_code_lines = getsourcelines(function)
                    experiment_id_line = [line for line in function_code_lines[0]
                                          if re.search(r'experiment_id = ', line)]

                    # if there was no experiment ID specified, it was either a
                    # a decorated test function (which means it wouldn't have an 'output'
                    # directory anyway) or it was a compare or prediction test function
                    # which are not a problem for various reasons.
                    if experiment_id_line:
                        experiment_id = eval(experiment_id_line[0].strip().split(' = ')[1])

                        # get the name of the source directory
                        source_line = [line for line in function_code_lines[0]
                                       if re.search(r'source = ', line)]
                        source = eval(source_line[0].strip().split(' = ')[1])
                        update_test_output(source,
                                           experiment_id,
                                           args.filename,
                                           args.outputs_dir)


if __name__ == '__main__':
    main()
