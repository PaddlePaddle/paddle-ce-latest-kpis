#!/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
cp -r -n ${models_dir}/PaddleNLP/language_model/. ./

cd data
if [ ! -d "simple-examples" ];then
ln -s ${data_path}/simple-examples simple-examples
fi
cd ..

ln -s ${models_dir}/PaddleNLP/shared_modules ../shared_modules
./.run_ce.sh
rm -rf ../shared_modules

