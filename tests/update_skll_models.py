#!/usr/bin/env python

"""
Update SKLL models in tests.

This script updates the SKLL model files to ensure they are compatible with the
current version. It simply iterates through all rsmpredict tests under the
`data/experiments` directory, finds the model files, loads them, and saves
them again.

IMPORTANT:
- You must run this from the `tests/` directory.
- Running this script will overwrite the original test data.
- Before running this script, make sure that the outputs work as expected.

:author: Anastassia Loukina
:author: Nitin Madnani

:organization: ETS
"""

import glob
from os import getcwd, remove
from os.path import dirname, join

from skll.learner import Learner

TEST_DIR = getcwd()


def update_model(model_file):
    """Read in the model file and save it again."""
    model_dir = dirname(model_file)

    # get the list of current files so that we can
    # remove them later to ensure there are no stranded
    # .npy files
    npy_files = glob.glob(join(model_dir, '*.npy'))

    # now load the SKLL model
    model = Learner.from_file(model_file)

    # delete the existing npy files. The model file will get overwritten,
    # but we do not know the exact number of current .npy files.
    for npy_file in npy_files:
        remove(npy_file)

    model.save(model_file)


def main():  # noqa: D103
    model_files = glob.glob(join(TEST_DIR, 'data', 'experiments',
                                 '*predict*', 'existing_experiment',
                                 'output', "*.model"))
    for model_file in model_files:
        print(f"Processing {model_file}")
        update_model(model_file)


if __name__ == "__main__":
    main()
