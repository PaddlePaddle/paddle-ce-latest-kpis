#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
yum install libsndfile.x86_64 -y
rm -rf ${models_dir}/PaddleSpeech/DeepVoice3/.run_ce.sh
rm -rf ${models_dir}/PaddleSpeech/DeepVoice3/_ce.py
cp -r ${models_dir}/PaddleSpeech/DeepVoice3/. ./
python download.py
if [ -d "data" ];then rm -rf data
fi
ln -s ${dataset_path}/DeepVoice3/data data
./.run_ce.sh
