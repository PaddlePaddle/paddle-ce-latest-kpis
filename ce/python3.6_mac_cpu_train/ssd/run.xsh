#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r -n ${models_dir}/PaddleCV/ssd/. ./

cd data
if [ -d "pascalvoc" ];then
rm -rf pascalvoc
ln -s ${data_path}/pascalvoc pascalvoc
fi
cd ..

cd pretrained
if [ ! -d "ssd_mobilenet_v1_coco" ];then
ln -s ${data_path}/ssd_mobilenet_v1_coco ssd_mobilenet_v1_coco
fi
cd ..

./.run_ce.sh
