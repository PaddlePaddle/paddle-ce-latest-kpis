#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/PaddleCV/face_detection/.run_ce.sh
cp -r ${models_dir}/PaddleCV/face_detection/. ./
if [ -d "data" ];then rm -rf data
fi
ln -s ${dataset_path}/face_detection/data data

./.run_ce.sh
