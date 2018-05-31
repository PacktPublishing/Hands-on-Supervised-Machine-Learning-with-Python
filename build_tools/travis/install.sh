#!/bin/bash
# This script is meant to be called by the "install" step defined in
# .travis.yml. See http://docs.travis-ci.com/ for more details.
# The behavior of the script is controlled by environment variables defined
# in the .travis.yml in the top level folder of the project.

set -e

echo 'List files from cached directories'
echo 'pip:'
ls $HOME/.cache/pip

# for caching
export CC=/usr/lib/ccache/gcc
export CXX=/usr/lib/ccache/g++
# Useful for debugging how ccache is used
# export CCACHE_LOGFILE=/tmp/ccache.log
# ~60M is used by .ccache when compiling from scratch at the time of writing
ccache --max-size 100M --show-stats

# Deactivate the travis-provided virtual environment and setup a
# conda-based environment instead.
deactivate || echo "No virtualenv or condaenv to deactivate"

# install conda
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
MINICONDA_PATH=/home/travis/miniconda

# append the path, update conda
chmod +x miniconda.sh && ./miniconda.sh -b -p $MINICONDA_PATH
export PATH=$MINICONDA_PATH/bin:$PATH
conda update --yes conda

# Create the conda env and install the requirements
conda create -n testenv --yes python=${PYTHON_VERSION}
source activate testenv
pip install -r requirements.txt
pip install pytest pytest-cov coverage codecov

# set up the package
python setup.py install
