#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r -n ${models_dir}/PaddleCV/PaddleGAN/. ./

export PYTHONPATH=.;%PYTHONPATH%
cd dataset
if [ -d "coco" ];then
rm -rf coco
fi
if [ -d "voc" ];then
rm -rf voc
fi
ln -s ${data_path}/COCO17 coco
ln -s ${data_path}/pascalvoc voc
cd ..
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
# if cython is not installed
pip install Cython
# Install into global site-packages
make install
pip install tqdm
./.run_ce.sh
