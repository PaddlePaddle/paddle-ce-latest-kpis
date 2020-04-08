#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
echo "start copy model"
rm -rf ${models_dir}/PaddleSlim/.run_ce.sh
rm -rf ${models_dir}/PaddleSlim/_ce.py
cp -r ${models_dir}/PaddleSlim/. ./
echo "copy model end"
if [ -d "data" ];then rm -rf data
fi
if [ -d "pretrain" ];then rm -rf pretrain
fi
ln -s ${dataset_path}/slim/data data
ln -s ${dataset_path}/slim/slim_pretrain pretrain
cp ${dataset_path}/slim/sen_data/mobilenet_acc_top1_sensitive.data ./
ln -s ${dataset_path}/slim/class_pretrain ./classification/pretrain

./.run_ce.sh
