#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/PaddleRec/ssr/_ce.py
rm -rf ${models_dir}/PaddleRec/ssr/.run_ce.sh
cp -r ${models_dir}/PaddleRec/ssr/. ./
if [ -d 'train_big_data' ];then rm -rf train_big_data
fi
if [ -f 'vocab_big.txt' ];then rm -rf vocab_big.txt
fi
cp -r ${dataset_path}/ssr/* .

./.run_ce.sh
