#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r ${models_dir}/dygraph/lac/. ./
if [ -d "data" ];then rm -rf data
fi
ln -s ${dataset_path}/dygraph_lac/data data

./.run_ce.sh
