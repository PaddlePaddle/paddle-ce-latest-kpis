#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/PaddleNLP/dialogue_system/auto_dialogue_evaluation/.run_ce.sh
rm -rf ${models_dir}/PaddleNLP/dialogue_system/auto_dialogue_evaluation/_ce.py
cp -r ${models_dir}/PaddleNLP/dialogue_system/auto_dialogue_evaluation/. ./
if [ -d "ade/data" ];then rm -rf ade/data
fi
ln -s ${dataset_path}/auto_dialogue_evaluation/data ade/data
./.run_ce.sh
