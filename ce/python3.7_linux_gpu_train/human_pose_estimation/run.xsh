#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r ${models_dir}/PaddleCV/human_pose_estimation/. ./
if [ -d 'data' ];then rm -rf data
fi
mkdir data
ln -s ${dataset_path}/human_pose/MPII data/mpii
if [ -d 'pretrained' ];then rm -rf pretrained
fi
ln -s ${dataset_path}/human_pose/pretrained pretrained

./.run_ce.sh
