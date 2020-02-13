#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r -n ${models_dir}/PaddleCV/PaddleVideo/. ./

cd data/dataset
rm -rf  kinetics
ln -s ${data_path}/k400 kinetics
cd ../..
pip install wget
pip install h5py

./.run_ce.sh
