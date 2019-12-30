#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r ${models_dir}/PaddleNLP/PaddleTextGEN/variational_seq2seq/. ./
if [ -d "data" ];then rm -rf data
fi
ln -s ${dataset_path}/PaddleTextGEN/variational_seq2seq/data data

./.run_ce.sh
