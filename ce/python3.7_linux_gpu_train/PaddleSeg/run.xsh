#!/bin/bash
export seg_dir=$PWD/../../seg_repo
#copy models files
cp -r ${seg_dir}/. ./
if [ -d 'dataset/cityscapes' ];then rm -rf dataset/cityscapes
fi
ln -s ${dataset_path}/cityscape dataset/cityscapes
if [ -d 'pretrain' ];then rm -rf pretrain
fi
ln -s ${dataset_path}/paddleseg/pretrain pretrain
pip install -r requirements.txt
./.run_ce.sh
