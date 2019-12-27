#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r -n ${models_dir}/PaddleNLP/similarity_net/. ./

if [ ! -d "data" ];then
ln -s ${data_path}/similarity_net/data data
fi
if [ ! -d "model_files" ];then
ln -s ${data_path}/similarity_net/model_files model_files
fi
ln -s ${models_di}/PaddleNLP/models ../models

./.run_ce.sh
rm -rf ../modles
