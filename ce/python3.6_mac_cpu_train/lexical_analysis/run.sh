#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r -n ${models_dir}/PaddleNLP/lexical_analysis/. ./

if [ ! -d "data"];then
ln -s ${data_path}/lexical_analysis\data data
fi
if [ ! -d "pretrained"];then
ln -s ${data_path}/lexical_analysis\pretrained pretrained
fi
if [ ! -d "model_baseline"];then
ln -s ${data_path}/lexical_analysis\model_baseline model_baseline
fi
if [ ! -d "model_finetuned"];then
ln -s ${data_path}/lexical_analysis\model_finetuned model_finetuned
fi


ln -s ${models_dir}/PaddleNLP/models ../models
ln -s ${models_dir}/PaddleNLP/preprocess ../preprocess
./.run_ce.sh
rm -rf ../models
rm -rf ../preprocess
