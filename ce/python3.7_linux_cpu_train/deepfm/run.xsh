#!/bin/bash
# ce
export models_dir=$PWD/../../models_repo
#copy models files
cp -r ${models_dir}/PaddleRec/ctr/deepfm/* ./
if [ -d "data" ];then rm -rf data
fi
ln -s ${dataset_path}/deepfm data

./.run_ce.sh