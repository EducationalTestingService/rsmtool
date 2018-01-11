#!/bin/bash

# Abort on failed things
set -ex
set -o pipefail

#### Setup environment ####
export PATH=/opt/python/conda_default/bin:$PATH

# Set conda settings
conda config --force --remove-key channels || true
conda config --force --set always_yes yes --set changeps1 no || true
conda config --force --remove-key envs_dirs || true

# remove already existing environment directory
rm -rf $PWD/envs/rsmtool-refactor

# Create conda environment
mkdir -p $PWD/envs

# Make sure we clean up if there's an error
trap 'conda clean --lock' KILL INT TERM ERR
conda install -m -c desilinguist -p $PWD/envs/rsmtool-refactor --yes --copy --file rsmtool/conda_requirements.txt

# Don't call conda clean --lock if anything other than conda install dies
trap - KILL INT TERM ERR

# Install rsmtool Python packages
source activate $PWD/envs/rsmtool-refactor
pushd $PWD/rsmtool
pip install -e .
popd
source deactivate

# Create files with path to conda environment and to the current directory
echo $PWD/envs/rsmtool-refactor > conda_path
