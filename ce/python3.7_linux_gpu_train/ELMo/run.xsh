#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/PaddleNLP/pretrain_language_models/ELMo/.run_ce.sh
cp -r ${models_dir}/PaddleNLP/pretrain_language_models/ELMo/. ./
ln -s ${dataset_path}/ELMo/data data

./.run_ce.sh
