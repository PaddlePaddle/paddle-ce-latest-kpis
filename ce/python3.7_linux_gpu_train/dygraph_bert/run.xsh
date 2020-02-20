#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r ${models_dir}/dygraph/bert/. ./
if [ -d "data" ];then rm -rf data
fi
ln -s ${dataset_path}/dygraph_bert/data data

./.run_ce.sh
