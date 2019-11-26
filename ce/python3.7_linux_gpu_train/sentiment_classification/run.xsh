#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/PaddleNLP/sentiment_classification/.run_ce.sh
cp -r ${models_dir}/PaddleNLP/sentiment_classification/. ./
if [ -d "senta_data" ];then rm -rf senta_data
fi
ln -s ${dataset_path}/sentiment_classification/senta_data senta_data

./.run_ce.sh
