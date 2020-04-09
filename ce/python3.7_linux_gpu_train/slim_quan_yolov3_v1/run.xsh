#!/bin/bash
export detection_dir=$PWD/../../detection_repo
#copy models files
cp -r ${detection_dir}/. ./
if [ -d "dataset" ];then rm -rf dataset
fi
ln -s ${dataset_path}/yolov3/dataset dataset

./.run_ce.sh
