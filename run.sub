#!/bin/bash
# Following the example from http://chtc.cs.wisc.edu/conda-installation.shtml
# except here we download the installer instead of transferring it
# Download a specific version of Miniconda instead of latest to improve
# reproducibility

# unzip 
tar zxf cifar-100-python.tar.gz

export HOME=$PWD
wget -q https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh -O miniconda.sh
sh miniconda.sh -b -p $HOME/miniconda3
rm miniconda.sh
export PATH=$HOME/miniconda3/bin:$PATH

# Set up conda
source $HOME/miniconda3/etc/profile.d/conda.sh
hash -r
conda config --set always_yes yes --set changeps1 no

# Install packages specified in the environment file
conda env create -f environment.yml

# Activate the environment and log all packages that were installed
conda activate attenreplace

python main.py

