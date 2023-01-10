#!/usr/bin/env python
"""
Copy over JSON configuration files from tests and rename them.

This utility script copies over json files from rsmextra/rsmtool tests and
rename them so that they all have unique names which match the experiment ids
with the tool name appended to the end

:author: Anastassia Loukina (aloukina@ets.org)
:author: Nitin Madnani (nmadnani@ets.org)
:author: Jeremy Biggs (jbiggs@ets.org)
:organization: ETS
"""


import argparse
import glob
import json
import re
from os import getcwd
from os.path import abspath, basename, dirname, exists, join

from rsmtool.input import parse_json_with_comments

PATH_FIELDS = ['train_file',
               'test_file',
               'features',
               'input_features_file',
               'experiment_dir',
               'experiment_dir_old',
               'experiment_dir_new',
               'predictions_file',
               'scale_with']

# list of tests to skip as they test distributional properties or
# or deprecated config file format
SKIP_TESTS = ['lr-with-subgroup-as-feature-name',
              'lr-with-sc2-as-feature-name',
              'lr-with-sc1-as-feature-name',
              'lr-with-repeated-ids',
              'lr-with-only-one-fully-numeric-feature',
              'lr-with-none-flagged',
              'lr-with-missing-length-values',
              'lr-with-length-zero-sd',
              'lr-with-length-as-feature-name',
              'lr-with-length-and-feature',
              'lr-with-large-integer-value',
              'lr-with-duplicate-feature-names',
              'lr-with-defaults-as-extra-columns',
              'lr-with-all-non-numeric-scores',
              'lr-subgroups-with-edge-cases',
              'lr-rsmtool-rsmpredict',
              'lr-missing-values',
              'empwtdropneg',  # this is deprecated
              'lr-predict-with-repeated-ids',
              'lr-predict-missing-values',
              'lr-predict-missing-postprocessing-file',
              'lr-predict-missing-model-file',
              'lr-predict-missing-feature-file',
              'lr-predict-illegal-transformations',
              'lr-eval-with-repeated-ids',
              'lr-eval-with-missing-scores',
              'lr-eval-with-missing-h2-column',
              'lr-eval-with-missing-data',
              'lr-eval-with-missing-candidate-column',
              'lr-eval-with-all-non-numeric-scores',
              'lr-eval-with-all-non-numeric-machine-scores',
              'lr-eval-tool-compare',
              'lr-eval-self-compare',
              'lr-different-compare',
              'linearsvr-self-compare'
              ]


def copy_jsons(source_dir, target_dir):
    """Find JSON files in the ``source_dir`` and copy to ``target_dir``."""
    dir_content = glob.glob(join(source_dir, '**/*.json'))

    # iterate over jsons and copy as applicable
    for filename in dir_content:

        experiment_dir = basename(dirname(filename))

        # skip the experiments not applicable to rsmapp
        if experiment_dir in SKIP_TESTS:
            continue

        # we are not interested in feature json files
        if filename == 'features.json':
            continue

        json_obj = parse_json_with_comments(filename)

        # we are going to append the tool name to the experiment id/file name
        # to make it easier for the tester so we need to identify the tool
        if 'train_file' in json_obj:
            tool = 'rsmtool'
        elif 'input_features_file' in json_obj:
            tool = 'rsmpredict'
        elif 'system_score_column' in json_obj:
            tool = 'rsmeval'
        elif 'experiment_id_old' in json_obj:
            tool = 'rsmcompare'
        # if it's none of this, we don't want to do anything
        else:
            continue

        # change the id and the file name to the directory name since these are more
        # explicit about the purpose of the test
        if not tool == 'rsmcompare':
            new_id = f'{experiment_dir}_{tool}'
            json_obj['experiment_id'] = new_id
            output_fname = join(target_dir, f'{new_id}.json')

        # for rsmcompare we replace the first id with the directory name
        if tool == 'rsmcompare':
            new_old_id = f'{experiment_dir}_{tool}'
            json_obj['experiment_id_old'] = new_old_id
            output_fname = join(target_dir, f"{new_old_id}_vs_{json_obj['experiment_id_new']}.json")

        # convert paths in reference files to absolute paths and replace
        # My Documents with Documents
        for field in PATH_FIELDS:
            if (field in json_obj and
                    not json_obj[field] is not None and
                    not json_obj[field] == 'asis'):
                new_ref_path = abspath(join(source_dir,
                                            experiment_dir,
                                            json_obj[field]))
                new_ref_path = re.sub("My Documents", "Documents", new_ref_path)
                json_obj[field] = new_ref_path

        if exists(output_fname):
            print(f"WARNING: {basename(output_fname)} exists and will be overwritten")
        with open(output_fname, 'w') as outfile:
            json.dump(json_obj, outfile, indent=4, separators=(',', ': '))


def main():  # noqa: D103
    # set up an argument parser
    parser = argparse.ArgumentParser(prog='copy_jsons.py')
    parser.add_argument(dest='source_dir',
                        help="Parent directory containing .json files")
    parser.add_argument(dest='target_dir',
                        help="Target directory", default=getcwd(), nargs='?')
    # parse given command line arguments
    args = parser.parse_args()
    copy_jsons(args.source_dir, args.target_dir)


if __name__ == "__main__":

    main()
