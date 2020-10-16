#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r ${models_dir}/dygraph/tsm/. ./

if [ -f "data/dataset/kinetics/*.list" ];then rm -rf data/dataset/kinetics/*.list
fi
ln -s ${dataset_path}/ResNet50_pretrained ResNet50_pretrained
ln -s ${dataset_path}/video/kinetics/train.list data/dataset/kinetics/train.list
ln -s ${dataset_path}/video/kinetics/val.list data/dataset/kinetics/val.list 
wget https://paddlemodels.bj.bcebos.com/video_classification/ResNet50_pretrained.tar.gz
tar -xvf ResNet50_pretrained.tar.gz
./.run_ce.sh
