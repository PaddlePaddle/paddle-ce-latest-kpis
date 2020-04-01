#!/bin/bash
export detection_dir=$PWD/../../detection_repo
#copy models files
cp -r ${detection_dir}/. ./
if [ -d 'dataset/coco' ];then rm -rf dataset/coco
fi
ln -s ${dataset_path}/yolov3/dataset/coco dataset/coco
if [ -d 'dataset/voc' ];then rm -rf dataset/voc
fi
ln -s ${dataset_path}/pascalvoc dataset/voc
if [ -d 'dataset/wider_face' ];then rm -rf dataset/wider_face
ln -s ${dataset_path}/wider_face dataset/wider_face
./.run_ce.sh
