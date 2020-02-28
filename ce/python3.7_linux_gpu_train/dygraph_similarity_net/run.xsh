#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r ${models_dir}/dygraph/similarity_net/. ./
if [ -d "dataset" ];then rm -rf dataset
fi
ln -s ${dataset_path}/dygraph_similarity_net/dataset

./.run_ce.sh
