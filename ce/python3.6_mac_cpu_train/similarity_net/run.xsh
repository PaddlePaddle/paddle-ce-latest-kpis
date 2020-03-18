#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r -n ${models_dir}/PaddleNLP/similarity_net/. ./

if [ ! -d "data" ];then
ln -s ${data_path}/similarity_net/data data
fi

ln -s ${models_dir}/PaddleNLP/models ../shared_modules

./.run_ce.sh
rm -rf ../shared_modules
