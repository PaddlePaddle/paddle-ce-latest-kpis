#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/PaddleCV/rcnn/.run_ce.sh
cp -r ${models_dir}/PaddleCV/rcnn/. ./
if [ -d "dataset" ];then rm -rf dataset
fi
if [ -d "imagenet_resnet50_fusebn" ];then rm -rf imagenet_resnet50_fusebn
fi
ln -s ${dataset_path}/faster_rcnn/dataset dataset
ln -s ${dataset_path}/faster_rcnn/imagenet_resnet50_fusebn imagenet_resnet50_fusebn

./.run_ce.sh
