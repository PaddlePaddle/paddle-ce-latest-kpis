#!/bin/bash
export detection_dir=$PWD/../../detection_repo
pip install paddleslim
#copy PaddleDetection files
cp -r ${detection_dir}/. ./

if [ -d "dataset" ];then rm -rf dataset
fi
ln -s ${dataset_path}/yolov3/dataset dataset

./.run_ce.sh
