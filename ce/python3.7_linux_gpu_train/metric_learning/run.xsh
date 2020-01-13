#!/bin/bash

export models_dir=$PWD/../../models_repo
rm -rf ${models_dir}/PaddleCV/metric_learning/_ce.py
${models_dir}/PaddleCV/metric_learning/.run_ce.sh
cp -r ${models_dir}/PaddleCV/metric_learning/. ./ 
if [ -d "data/Stanford_Online_Products"];then rm -rf data/Stanford_Online_Products
fi
ln -s ${dataset_path}/Stanford_Online_Products data/Stanford_Online_Products

./.run_ce.sh
