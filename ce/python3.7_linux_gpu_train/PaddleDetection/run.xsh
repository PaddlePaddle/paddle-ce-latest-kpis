#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r ${models_dir}/PaddleCV/PaddleDetection/. ./
if [ -d 'dataset/coco' ];then rm -rf dataset/coco
fi
ln -s ${dataset_path}/yolov3/dataset/coco dataset/coco

./.run_ce.sh
