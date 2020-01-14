#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/PaddleCV/image_classification/.run_ce.sh
rm -rf ${models_dir}/PaddleCV/image_classification/_ce.py
cp -r ${models_dir}/PaddleCV/image_classification/. ./
if [ -d "data/ILSVRC2012" ];then rm -rf data/ILSVRC2012
fi
ln -s ${dataset_path}/ILSVRC2012_small data/ILSVRC2012
./.run_ce.sh
