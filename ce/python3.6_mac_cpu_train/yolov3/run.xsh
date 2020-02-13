#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r -n ${models_dir}/PaddleCV/yolov3/. ./

rm -rf weights
ln -s ${data_path}/yolov3/weights weights
rm -rf dataset
ln -s ${data_path}/yolov3/dataset dataset

./.run_ce.sh
