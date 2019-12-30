#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/PaddleNLP/similarity_net/.run_ce.sh
rm -rf ${models_dir}/PaddleNLP/similarity_net/_ce.py
cp -r ${models_dir}/PaddleNLP/similarity_net/. ./

if [ -d "data" ];then rm -rf data
fi
ln -s ${dataset_path}/similarity_net/data data
./.run_ce.sh
