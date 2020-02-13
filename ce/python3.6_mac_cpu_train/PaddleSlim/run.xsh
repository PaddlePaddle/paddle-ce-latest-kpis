#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r -n ${models_dir}/PaddleSlim/. ./
if [ ! -d "data" ];then
mkdir data
ln -s ${data_path}/ILSVRC2012 data/ILSVRC2012
fi
if [ ! -d "pretrain" ];then
ln -s ${data_path}/PaddleSlim pretrain
fi
if [ -d "checkpoints" ];then
rm -rf checkpoints
fi
./.run_ce.sh
