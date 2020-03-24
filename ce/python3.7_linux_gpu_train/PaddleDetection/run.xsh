#!/bin/bash
export detection_dir=$PWD/../../detection_repo
#copy models files
cp -r ${detection_dir}/. ./
if [ -d 'dataset/coco' ];then rm -rf dataset/coco
fi
ln -s ${dataset_path}/yolov3/dataset/coco dataset/coco

./.run_ce.sh
