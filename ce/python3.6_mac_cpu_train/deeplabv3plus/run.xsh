#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r -n ${models_dir}/PaddleCV/deeplabv3+/. ./

if [ -d "data" ];then
rm -rf data
fi
if [ -d "model" ];then
rm -rf model
fi

if [ ! -d "deeplabv3plus_gn_init" ];then
ln -s  ${data_path}/deeplabv3+/deeplabv3plus_gn_init deeplabv3plus_gn_init
fi
if [ ! -d "deeplabv3plus_gn_init" ];then
ln -s  ${data_path}/deeplabv3+/deeplabv3plus_gn deeplabv3plus_gn
fi

mkdir model
mkdir data
cd data
ln -s  ${data_path}/cityscape cityscape
cd ..

./.run_ce.sh
