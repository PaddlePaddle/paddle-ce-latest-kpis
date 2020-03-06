#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r ${models_dir}/dygraph/yolov3/. ./

if [ -d 'dataset' ];then rm -rf dataset
fi
ln -s ${dataset_path}/yolov3/dataset dataset
sh weights/download.sh
./.run_ce.sh
