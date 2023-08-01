#!/usr/bin/env python

"""
Update RSMTool models in tests.

This script updates the RSMTool model files to ensure they are compatible with
the current version. It simply iterates through all rsmpredict tests under the
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
from os import getcwd
from os.path import join

from rsmtool.modeler import Modeler

TEST_DIR = getcwd()


def update_model(model_file):
    """Read in the model file and save it again."""
    model = Modeler.load_from_file(model_file)
    model.save(model_file)


def main():  # noqa: D103
    model_files = (
        glob.glob(
            join(
                TEST_DIR,
                "data",
                "experiments",
                "*predict*",
                "existing_experiment",
                "output",
                "*.model",
            )
        )
        + glob.glob(
            join(
                TEST_DIR,
                "data",
                "experiments",
                "*explain*",
                "existing_experiment",
                "output",
                "*.model",
            )
        )
        + [join(TEST_DIR, "data", "files", "explain_svr.model")]
    )

    for model_file in model_files:
        print(f"Processing {model_file}")
        update_model(model_file)


if __name__ == "__main__":
    main()
