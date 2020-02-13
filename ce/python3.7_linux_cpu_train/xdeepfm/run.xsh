#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r ${models_dir}/PaddleRec/ctr/xdeepfm/* ./
if [ -d "data" ];then rm -rf data
fi
ln -s ${dataset_path}/xdeepfm data

./.run_ce.sh
