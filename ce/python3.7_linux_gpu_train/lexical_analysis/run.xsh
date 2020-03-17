#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/PaddleNLP/lexical_analysis/.run_ce.sh
rm -rf ${models_dir}/PaddleNLP/lexical_analysis/_ce.py
cp -r ${models_dir}/PaddleNLP/lexical_analysis/. ./
cd ..
if [ -d "shared_modules" ];then rm -rf shared_modules
fi
cp -r ${models_dir}/PaddleNLP/shared_modules .
cd lexical_analysis
if [ -d "data" ];then rm -rf data
fi
ln -s ${dataset_path}/lexical_analysis/data data

./.run_ce.sh
