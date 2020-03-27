#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/PaddleRec/tagspace/.run_ce.sh
rm -rf ${models_dir}/PaddleRec/tagspace/_ce.py
cp -r ${models_dir}/PaddleRec/tagspace/. ./
if [ -d 'train_big_data' ];then rm -rf train_big_data
fi
if [ -f 'big_vocab_tag.txt' ];then rm -rf big_vocab_tag.txt
fi
if [ -f 'big_vocab_text.txt' ];then rm -rf big_vocab_text.txt
fi
cp -r ${dataset_path}/tagspace/* .
./.run_ce.sh
