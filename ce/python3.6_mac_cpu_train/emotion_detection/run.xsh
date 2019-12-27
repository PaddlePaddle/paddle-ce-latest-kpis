#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r -n ${models_dir}/PaddleNLP/emotion_detection/. ./

if [ ! -d "data" ];then
ln -s ${data_path}/emotion_detection/data data
fi
if [ ! -d "models" ];then
ln -s ${data_path}/emotion_detection/models models
fi
ln -s ${models_di}/PaddleNLP/models ../models

./.run_ce.sh
rm -rf ../modles
