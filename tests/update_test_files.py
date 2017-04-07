#!/usr/bin/env python

"""
This script is designed to update the test files.
It takes the suffix of the output file and updates the test data using the test_outputs.
Note that this will overwrite the original test data.
Before running this script, make sure that the outputs work as expected

:author: Anastassia Loukina
:author: Nitin Madnani
:date: June
"""

import argparse
import inspect
import re
import shutil


from ast import literal_eval as eval
from importlib.machinery import SourceFileLoader
from os.path import dirname, exists, join

_MY_PATH = dirname(__file__)


def main():
    # set up an argument parser
    parser = argparse.ArgumentParser(prog='update_test_files.py')
    parser.add_argument('suffix', help="The suffix of the output file")
    parser.add_argument('outputs_dir', help="The path to test_outputs")

    # parse given command line arguments
    args = parser.parse_args()

    suffix = args.suffix
    out_dir = args.outputs_dir

    # import the test_experiment.py test module using SourceFileLoader
    # so that it could also be used with tests in RSMExtra provided
    # they folllow the same structure.
    # see http://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
    # for Python 3.5 solution
    test_experiment = SourceFileLoader('test_experiment', join(_MY_PATH, 'test_experiment.py')).load_module()

    # iterate over all the members and focus on only the experiment functions
    for member in inspect.getmembers(test_experiment):
        if inspect.isfunction(member[1]) and member[0].startswith('test_run_experiment'):
            function = member[1]
            # get the experiment id and the source for each test function
            function_code_lines = inspect.getsourcelines(function)
            experiment_id_line = [line for line in function_code_lines[0]
                                  if re.search(r'experiment_id = ', line)]

            # if there was no experiment ID specified, it was either a
            # a decorated test function (which means it wouldn't have an 'output'
            # directory anyway) or it was a compare or prediction test function
            # which are not a problem for various reasons.
            if experiment_id_line:
                experiment_id_in_test = eval(experiment_id_line[0].strip().split(' = ')[1])

                # get the name of the source directory
                source_line = [line for line in function_code_lines[0]
                                      if re.search(r'source = ', line)]
                source = eval(source_line[0].strip().split(' = ')[1])
                print(source)

                # check if the specified output file already exists in test data

                test_file = '{}/data/experiments/{}/output/{}_{}'.format(_MY_PATH, source, experiment_id_in_test, suffix)
                output_file = '{}/{}/output/{}_{}'.format(out_dir, source, experiment_id_in_test, suffix)

                if exists(output_file):
                    print('Copying {} to {}'.format(output_file, test_file))
                    shutil.copy(output_file, test_file)
                else:
                    print('{} does not exist'.format(output_file))

if __name__ == '__main__':
    main()
