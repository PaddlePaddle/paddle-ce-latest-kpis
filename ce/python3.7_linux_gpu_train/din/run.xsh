#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/PaddleRec/din/.run_ce.sh
rm -rf ${models_dir}/PaddleRec/din/_ce.py
cp -r ${models_dir}/PaddleRec/din/. ./
if [ -d "data" ];then rm -rf data
fi
cp -r ${dataset_path}/din/data .
cp -r ${dataset_path}/din/din_amazon .
./.run_ce.sh
