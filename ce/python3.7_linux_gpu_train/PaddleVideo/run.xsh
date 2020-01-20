#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/PaddleCV/PaddleVideo/.run_ce.sh
rm -rf ${models_dir}/PaddleCV/PaddleVideo/_ce.py
cp -r ${models_dir}/PaddleCV/PaddleVideo/. ./
if [ -f "data/dataset/kinetics/*.list" ];then rm -rf data/dataset/kinetics/*.list
fi
ln -s ${dataset_path}/video/kinetics/train.list data/dataset/kinetics/train.list
ln -s ${dataset_path}/video/kinetics/val.list data/dataset/kinetics/val.list
if [ -f "data/dataset/youtube8m/*.list" ];then rm -rf data/dataset/youtube8m/*.list
fi
ln -s ${dataset_path}/video/youtube/train.list data/dataset/youtube8m/train.list
ln -s ${dataset_path}/video/youtube/val.list data/dataset/youtube8m/val.list
if [ -f "data/dataset/nonlocal/*.list" ];then rm -rf data/dataset/nonlocal/*.list
fi
ln -s ${dataset_path}/video/nonlocal/trainlist.txt data/dataset/nonlocal/trainlist.txt
ln -s ${dataset_path}/video/nonlocal/vallist.txt data/dataset/nonlocal/vallist.txt
if [ -d "data/dataset/ctcn" ];then rm -rf data/dataset/ctcn
fi
ln -s ${dataset_path}/video/ctcn data/dataset/ctcn
./.run_ce.sh
