#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/PaddleCV/gan/_ce.py
rm -rf ${models_dir}/PaddleCV/gan/.run_ce.sh
cp -r ${models_dir}/PaddleCV/gan/. ./
if [ -d "data" ];then rm -rf data
fi
ln -s ${dataset_path}/gan data
if [ -d "VGG19_pretrained" ];then rm -rf VGG19_pretrained
fi
ln -s ${dataset_path}/gan/VGG19_pretrained VGG19_pretrained

./.run_ce.sh
