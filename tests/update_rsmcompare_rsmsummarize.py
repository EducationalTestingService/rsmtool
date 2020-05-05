#!/usr/bin/env python

"""
This script is designed to update the input files for
rsmcompare and rsmsummarize. It assumes that the data we used in these
tests is the same data we use for rsmtool and rsmeval tests.

- The script locates the rsmsummarize/rsmcompare files under the test outputs directory.
- The script compares them to the files in rsmcompare/rsmsummarize tests
- New and changed files under are copied over to the tests.
- Old files in the tests are deleted.
- Files that are already in the test and have not changed are left alone.

The script prints a log detailing the changes made for each experiment test.


:author: Anastassia Loukina
:author: Nitin Madnani

:organization: ETS
"""

import argparse
import re
import sys
import glob

from pathlib import Path

from shutil import copyfile, copytree, rmtree

def main():
    # set up an argument parser
    parser = argparse.ArgumentParser(prog='update_test_inputs.py')
    parser.add_argument('--tests',
                        dest='tests_dir',
                        required=True,
                        help="The path to the existing RSMTool tests directory")
    parser.add_argument('--outputs',
                        dest='outputs_dir',
                        required=True,
                        help="The path to the directory containing the updated test "
                             "outputs (usually `test_outputs`)")

    # parse given command line arguments
    args = parser.parse_args()

    # print out a reminder that the user should have run the test suite
    run_test_suite = input('Have you already run the whole test suite? (y/n): ')
    if run_test_suite == 'n':
        print('Please run the whole test suite using '
              '`nosetests --nologcapture` before running this script.')
        sys.exit(0)
    elif run_test_suite != 'y':
        print('Invalid answer. Exiting.')
        sys.exit(1)
    else:
        print()

    test_dir = Path(args.tests_dir)
    outputs_dir = Path(args.outputs_dir)

    # iterate over the given tests directory and find all directories
    # that contain another directory with 'output' folder

    for tool in ['summary', 'compare']:
        tests_with_input = test_dir.glob('data/experiments/'
                                         '*{}*/*/output'.format(tool))

        for output_dir in tests_with_input:
            input_test_dir = output_dir.parent
            source_name = input_test_dir.name
            output_dir = outputs_dir / source_name
            # check that this is a real test
            if tool=='summary' and not (input_test_dir.parent / 'output').exists():
                print("Skipping {}".format(input_test_dir.parent))
                continue
            elif not output_dir.exists():
                print("No new data found for {}".format(input_test_dir))
            else:
                #print("Updating input data for {}".format(input_test_dir))
                rmtree(input_test_dir / 'output')
                copytree(output_dir / 'output',
                                input_test_dir/ 'output')
                if tool == 'compare':
                    rmtree(input_test_dir / 'figure')
                    copytree(output_dir / 'figure',
                                    input_test_dir/ 'figure')


if __name__ == '__main__':
    main()
