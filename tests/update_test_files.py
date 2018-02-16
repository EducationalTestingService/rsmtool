#!/usr/bin/env python

"""
This script is designed to update the expected test output.
It assumes that you have already run nosetests and ran the entire test suite.
By doing so, the output has been generated under the given outputs directory.
and that is what will be used to generate the new expected output under `tests/data/experiments`.

#############################################################################################
# IMPORTANT: DO NOT RUN THIS SCRIPT BEFORE RUNNING THE TEST SUITE OR IT WILL BE DISASTROUS. #
#############################################################################################

The scripts works as as follows. For each experiment test:
- The script locates the output under the given outputs directory.
- New and changed files under `test_outputs` are copied over to the expected test output location.
- Old files in the expected test output that DO NOT exist under `test_outputs` are deleted.
- Files that are already in the expected test output and have not changed are left alone.

The script prints a log detailing the changes made for each experiment test.

:author: Anastassia Loukina
:author: Nitin Madnani
:author: Jeremy Biggs
:date: Feburary 2018
"""

import argparse
import re
import sys

from ast import literal_eval as eval
from filecmp import clear_cache, dircmp
from importlib.machinery import SourceFileLoader
from inspect import getmembers, getsourcelines, isfunction
from os import remove
from os.path import join
from pathlib import Path
from shutil import copyfile

from rsmtool.test_utils import check_file_output

_MY_PATH = Path(__file__).parent


def is_skll_excluded_file(filename):

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    return suffix == '.model' or \
        suffix == '.npy'or \
        stem.endswith('_postprocessing_params') or \
        stem.endswith('_eval') or \
        stem.endswith('_eval_short') or \
        stem.endswith('_confMatrix') or \
        stem.endswith('_pred_train') or \
        stem.endswith('_pred_processed') or \
        stem.endswith('_score_dist')


def update_output(source, outputs_dir, skll=False):

    # locate the updated outputs for the experiment under the given
    # outputs directory and also locate the existing experiment outputs
    updated_output_path = outputs_dir / source / "output"
    existing_output_path = _MY_PATH / "data" / "experiments" / source / "output"

    # make sure both the directories exist
    try:
        assert updated_output_path.exists()
    except AssertionError:
        sys.stderr.write("\nError: the given directory with updated "
                         "test outputs does not contain the outputs for "
                         "the \"{}\" experiment.\n".format(source))
        sys.exit(1)

    try:
        assert existing_output_path.exists()
    except AssertionError:
        sys.stderr.write("\nNo existing output for \"{}\". "
                         "Creating directory ...\n".format(source))
        existing_output_path.mkdir(parents=True)

    # get a comparison betwen the two directories
    dir_comparison = dircmp(updated_output_path, existing_output_path)

    # if no output was found in the updated outputs directory, that's a
    # likely to be a problem
    if not dir_comparison.left_list:
        sys.stderr.write("\nError: no updated outputs found for \"{}\".\n".format(source))
        sys.exit(1)

    # first delete the files that only exist in the existing output directory
    # since those are likely old files from old versions that we do not need
    existing_output_only_files = dir_comparison.right_only
    for file in existing_output_only_files:
        remove(existing_output_path / file)

    # Next find all the NEW files in the updated outputs but do not include
    # the config JSON files or the evaluation/prediction/model files for SKLL
    new_files = dir_comparison.left_only
    new_files = [f for f in new_files if not f.endswith('_rsmtool.json') and not f.endswith('_rsmeval.json')]
    if skll:
        new_files = [f for f in new_files if not is_skll_excluded_file(f)]

    # next we get the files that have changed and try to figure out if they
    # have actually changed beyond a tolerance level that we care about for
    # tests. To do this, we run the same function that we use when comparing
    # the files in the actual test. However, for non-tabular files, we just
    # assume that they have really changed since we have no easy wa to compare.
    changed_files = dir_comparison.diff_files
    really_changed_files = []
    for changed_file in changed_files:
        include_file = True
        updated_output_filepath = updated_output_path / changed_file
        existing_output_filepath = existing_output_path / changed_file
        file_format = updated_output_filepath.suffix.lstrip('.')
        if file_format in ['csv', 'tsv', 'xlsx']:
            try:
                check_file_output(str(updated_output_filepath),
                                  str(existing_output_filepath),
                                  file_format=file_format)
            except AssertionError:
                pass
            else:
                include_file = False

        if include_file:
            really_changed_files.append(changed_file)

    # Copy over the new files as well as the really changed files
    new_or_changed_files = new_files + really_changed_files
    for file in new_or_changed_files:
        copyfile(updated_output_path / file, existing_output_path / file)

    # return lists of files we deleted and updated
    return sorted(existing_output_only_files), sorted(new_or_changed_files)


def main():
    # set up an argument parser
    parser = argparse.ArgumentParser(prog='update_test_files.py')
    parser.add_argument('outputs_dir', help="The path to the directory containing the updated test outputs")

    # parse given command line arguments
    args = parser.parse_args()

    # invalidate the filecmp cache so that it starts fresh
    clear_cache()

    # print out a reminder that the user should have run the test suite
    run_test_suite = input('Have you already run the whole test suite? (y/n): ')
    if run_test_suite == 'n':
        print('Please run the whole test suite using '
              '`nosetests --nologcapture` before running this script.')
        sys.exit(0)
    elif run_test_suite != 'y':
        print('Invalid answer. Exiting.')
        sys.exit(1)

    # import the test_experiment.py test module using SourceFileLoader
    # so that it could also be used with tests in RSMExtra provided
    # they folllow the same structure.
    # see http://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
    # for Python 3.5 solution
    for test_suffix in ['rsmtool_1', 'rsmtool_2', 'rsmtool_3', 'rsmtool_4',
                        'rsmeval', 'rsmpredict', 'rsmsummarize']:
        test_module_path = join(_MY_PATH, 'test_experiment_{}.py'.format(test_suffix))
        test_module = SourceFileLoader('test_experiment', test_module_path).load_module()

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
                        source = param.args[0]
                        skll = param.kwargs.get('skll', False)

                        deleted, updated = update_output(source,
                                                         Path(args.outputs_dir),
                                                         skll=skll)
                        if len(deleted) > 0 or len(updated) > 0:
                            print('{}: '.format(source))
                            print('  - {} deleted: {}'.format(len(deleted), deleted))
                            print('  - {} added/updated: {}'.format(len(updated), updated))

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

                        # get the name of the source directory
                        source_line = [line for line in function_code_lines[0]
                                       if re.search(r'source = ', line)]
                        source = eval(source_line[0].strip().split(' = ')[1])
                        deleted, updated = update_output(source, Path(args.outputs_dir))
                        if len(deleted) > 0 or len(updated) > 0:
                            print('{}: '.format(source))
                            print('  - {} deleted: {}'.format(len(deleted), deleted))
                            print('  - {} added/updated: {}'.format(len(updated), updated))


if __name__ == '__main__':
    main()
