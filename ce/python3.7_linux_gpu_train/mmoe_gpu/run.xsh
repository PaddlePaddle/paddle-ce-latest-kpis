#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r ${models_dir}/PaddleRec/multi-task/MMoE/* ./
ln -s ${dataset_path}/mmoe/train_data train_data
ln -s ${dataset_path}/mmoe/test_data test_data
# train
pip install -r requirements.txt
./.run_ce.sh
