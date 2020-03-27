#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/PaddleNLP/emotion_detection/.run_ce.sh
rm -rf ${models_dir}/PaddleNLP/emotion_detection/_ce.py
cp -r ${models_dir}/PaddleNLP/emotion_detection/. ./
ln -s ${dataset_path}/emotion_detection/data data
./.run_ce.sh
