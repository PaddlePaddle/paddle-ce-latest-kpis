#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r -n ${models_dir}/PaddleRec/ctr/dnn/. ./
rm -rf data
ln -s ${data_path}/ctr/dnn data
pip install -r requirements.txt
./.run_ce.sh
