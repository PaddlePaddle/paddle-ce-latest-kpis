#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/PaddleRec/gnn/.run_ce.sh
rm -rf ${models_dir}/PaddleRec/gnn/_ce.py
cp -r ${models_dir}/PaddleRec/gnn/. ./
if [ -d "data" ];then rm -rf data
fi
ln -s ${dataset_path}/gnn/data data
./.run_ce.sh
