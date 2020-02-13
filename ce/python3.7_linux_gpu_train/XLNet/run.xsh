#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/PaddleNLP/PaddleLARK/XLNet/_ce.py
rm -rf ${models_dir}/PaddleNLP/PaddleLARK/XLNet/.run_ce.sh
cp -r ${models_dir}/PaddleNLP/PaddleLARK/XLNet/. ./
if [ -d 'data' ];then rm -rf data
fi
if [ -d 'xlnet_cased_L-12_H-768_A-12' ];then rm -rf xlnet_cased_L-12_H-768_A-12
fi
ln -s ${dataset_path}/XLNet/data data
ln -s ${dataset_path}/XLNet/xlnet_cased_L-12_H-768_A-12 xlnet_cased_L-12_H-768_A-12
./.run_ce.sh
