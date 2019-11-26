#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/PaddleNLP/PaddleMT/transformer/.run_ce.sh
cp -r ${models_dir}/PaddleNLP/PaddleMT/transformer/. ./
if [ -d "dataset" ];then rm -rf dataset
fi
ln -s ${dataset_path}/transformer/dataset dataset

./.run_ce.sh
