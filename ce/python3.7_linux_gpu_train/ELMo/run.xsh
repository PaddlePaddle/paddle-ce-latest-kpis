#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/PaddleNLP/PaddleLARK/ELMo/.run_ce.sh
cp -r ${models_dir}/PaddleNLP/PaddleLARK/ELMo/. ./
ln -s ${dataset_path}/ELMo/data data

./.run_ce.sh
