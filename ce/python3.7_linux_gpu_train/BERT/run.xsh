#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r ${models_dir}/PaddleNLP/PaddleLARK/BERT/. ./
if [ -d "data" ];then rm -rf data
fi
if [ -d "pretrain_model" ];then rm -rf pretrain_model
fi
ln -s ${dataset_path}/BERT/data data
ln -s ${dataset_path}/BERT/pretrain_model pretrain_model

./.run_ce.sh
