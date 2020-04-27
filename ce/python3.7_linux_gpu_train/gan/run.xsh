#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/PaddleCV/gan/_ce.py
rm -rf ${models_dir}/PaddleCV/gan/.run_ce.sh
cp -r ${models_dir}/PaddleCV/gan/. ./
if [ -d "data" ];then rm -rf data
fi
ln -s ${dataset_path}/gan data

./.run_ce.sh
