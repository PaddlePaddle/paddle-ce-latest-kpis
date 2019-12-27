#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r -n ${models_dir}/PaddleCV/image_classification/. ./
cd data
if [ -d "ILSVRC2012" ];then
rm -rf ILSVRC2012
ln -s ${data_path}/ILSVRC2012 ILSVRC2012
fi
cd ..
ln -s ${data_path}/ILSVRC2012/ResNet50_pretrained ResNet50_pretrained
./.run_ce.sh
