#!/usr/bin/env python

"""
This script is designed to update the sci-kit learn model files to ensure they
are compatible with the current version.
The script goes through all tests in the data/experiments directory, finds the model files and
saves them again.
Note that this will overwrite the original test data.
Before running this script, make sure that the outputs work as expected

:author: Anastassia Loukina
:author: Nitin Madnani
:date: February 2017
"""

import glob

from os import remove
from os.path import join, dirname
from skll import Learner

TEST_DIR = dirname(__file__)

def update_model(model_file):
    ''' Read in the model file and save it again'''

    model_dir = dirname(model_file)

    # get the list of current files so that we can
    #remove them later to ensure there are no stranded .npy

    npy_files = glob.glob(join(model_dir, '*.npy'))

    # now load the SKLL model
    model = Learner.from_file(model_file)

    # delete the existing npy files. The model file will get overwritten,
    # but we do not know the exact number of current .npy files.
    for npy_file in npy_files:
        remove(npy_file)

    model.save(model_file)


def main():
    model_files = glob.glob(join(TEST_DIR, 'data', 'experiments',
                                 '*predict*', 'existing_experiment',
                                 'output', "*.model"))
    for model_file in model_files:
        print("Processing {}".format(model_file))
        update_model(model_file)


if __name__ == "__main__":
    main()
