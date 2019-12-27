#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r -n ${models_dir}/PaddleCV/metric_learning/. ./
cd data
if [ -d "Stanford_Online_Products" ];then
rm -rf Stanford_Online_Products
ln -s ${data_path}/Stanford_Online_Products Stanford_Online_Products
fi
cd ..
./.run_ce.sh
