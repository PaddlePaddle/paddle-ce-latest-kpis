#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r -n ${models_dir}/PaddleCV/gan/. ./
if [ ! -d "data" ];then
ln -s ${data_path}/gan data
fi
./.run_ce.sh
