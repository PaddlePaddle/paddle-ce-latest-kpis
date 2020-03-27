#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r ${models_dir}/PaddleRec/word2vec/. ./
if [ -f 'data' ];then rm -rf data
fi
cp -r ${dataset_path}/word2vec data
./.run_ce.sh
