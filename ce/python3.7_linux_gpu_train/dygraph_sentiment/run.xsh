#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/dygraph/sentiment/_ce.py
rm -rf ${models_dir}/dygraph/sentiment/.run_ce.sh
cp -r ${models_dir}/dygraph/sentiment/. ./
if [ -d "senta_data" ];then rm -rf senta_data
fi
ln -s ${dataset_path}/dygraph_sentiment/senta_data senta_data

./.run_ce.sh
