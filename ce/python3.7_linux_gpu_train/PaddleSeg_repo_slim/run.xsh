#!/bin/bash
export seg_dir=$PWD/../../seg_repo
export slim_dir=$PWD/../../slim_repo

cd ${slim_dir}
python setup.py install
cd -
#copy PaddleSeg files
cp -r ${seg_dir}/. ./
pip install -r requirements.txt
if [ -d "dataset/cityscapes" ];then rm -rf dataset/cityscapes
fi
ln -s ${dataset_path}/cityscape dataset/cityscapes

if [ -d "pretrained_model" ];then rm -rf pretrained_model
fi
ln -s  ${dataset_path}/seg_slim/pretrained_model pretrained_model

ln -s  ${dataset_path}/seg_slim/test_img test_img

./.run_ce.sh
