#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/PaddleCV/PaddleVideo/.run_ce.sh
rm -rf ${models_dir}/PaddleCV/PaddleVideo/_ce.py
cp -r ${models_dir}/PaddleCV/PaddleVideo/. ./
if [ -f "data/dataset/kinetics/*.list" ];then rm -rf data/dataset/kinetics/*.list
fi
ln -s ${dataset_path}/video/train.list data/dataset/kinetics/train.list
ln -s ${dataset_path}/video/val.list data/dataset/kinetics/val.list
./.run_ce.sh
