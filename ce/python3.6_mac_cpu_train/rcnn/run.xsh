#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r -n ${models_dir}/PaddleCV/rcnn/. ./

cd dataset
if [ -d "coco" ];then
rm -rf coco
ln -s ${data_path}/COCO17 coco
fi
cd ..
if [ ! -d "imagenet_resnet50_fusebn" ];then
ln -s ${data_path}/rcnn/imagenet_resnet50_fusebn imagenet_resnet50_fusebn
fi

./.run_ce.sh
