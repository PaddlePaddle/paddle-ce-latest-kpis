#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r -n ${models_dir}/PaddleCV/face_detection/. ./

if [ -d "data" ];then
rm -rf data
ln -s  ${data_path}/face_detection/WIDERFACE data
fi

if [ ! -d "vgg_ilsvrc_16_fc_reduced" ];then
ln -s  ${data_path}/face_detection/vgg_ilsvrc_16_fc_reduced /vgg_ilsvrc_16_fc_reduced
fi
if [ ! -d "PyramidBox_WiderFace" ];then
ln -s  ${data_path}/face_detection/PyramidBox_WiderFace PyramidBox_WiderFace
fi


./.run_ce.sh
