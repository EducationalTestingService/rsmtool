#!/bin/bash

# Abort on failed things
set -ex
set -o pipefail

# create test results directory
mkdir -p rsmtool/testresults rsmtool/test_outputs

# Activate conda environment
export PATH=/opt/python/conda_default/bin:$PATH
CONDA_PATH=$(cat conda_path)
source activate $CONDA_PATH

# Set up IPYTHONDIR and run the tests
export IPYTHONDIR=/tmp/rsmtool_tests
mkdir -p $IPYTHONDIR
nosetests -v --with-xunit --xunit-file=rsmtool/testresults/all_tests.xml rsmtool/tests
rm -rf $IPYTHONDIR
