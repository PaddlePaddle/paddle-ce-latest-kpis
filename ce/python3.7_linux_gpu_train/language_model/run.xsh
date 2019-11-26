#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/PaddleNLP/language_model/.run_ce.sh
cp -r ${models_dir}/PaddleNLP/language_model/. ./
if [ -d "data" ];then rm -rf data
fi
cd ..
if [ -d "models" ];then rm -rf models
fi
cp -r ${models_dir}/PaddleNLP/models .
cd language_model
ln -s ${dataset_path}/nlp_language_model/data data

./.run_ce.sh
