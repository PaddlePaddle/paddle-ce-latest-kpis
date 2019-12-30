#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/PaddleCV/yolov3/.run_ce.sh
cp -r ${models_dir}/PaddleCV/yolov3/. ./
if [ -d 'dataset' ];then rm -rf dataset
fi
if [ -d 'weights' ];then rm -rf weights
fi
ln -s ${dataset_path}/yolov3/dataset dataset
ln -s ${dataset_path}/yolov3/weights weights

./.run_ce.sh
