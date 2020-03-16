#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/PaddleNLP/machine_translation/transformer/.run_ce.sh
cp -r ${models_dir}/PaddleNLP/machine_translation/transformer/. ./
if [ -d "dataset" ];then rm -rf dataset
fi
ln -s ${dataset_path}/transformer/dataset dataset

./.run_ce.sh
