#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/PaddleRec/ctr/dnn/.run_ce.sh
rm -rf ${models_dir}/PaddleRec/ctr/dnn/_ce.py
cp -r ${models_dir}/PaddleRec/ctr/dnn/. ./
if [ -d "data" ];then rm -rf data
fi
ln -s ${dataset_path}/ctr/data data
./.run_ce.sh
