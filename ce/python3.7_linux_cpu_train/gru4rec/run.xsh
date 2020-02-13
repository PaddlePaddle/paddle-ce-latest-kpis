#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/PaddleRec/gru4rec/.run_ce.sh
rm -rf ${models_dir}/PaddleRec/gru4rec/_ce.py
cp -r ${models_dir}/PaddleRec/gru4rec/. ./
if [ -d "train_big_data" ];then rm -rf train_big_data
fi
if [ -f "vocab_big.txt" ];then rm -rf vocab_big.txt
fi
cp -r ${dataset_path}/gru4rec/* .

./.run_ce.sh
