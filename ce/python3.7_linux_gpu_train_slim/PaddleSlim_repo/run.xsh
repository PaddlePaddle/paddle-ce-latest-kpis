#!/bin/bash
export slim_dir=$PWD/../../slim_repo
#copy PaddleSlim files
cp -r ${slim_dir}/. ./
pip install -r requirements.txt
python setup.py install
pip install parl
if [ -d "demo/data" ];then rm -rf demo/data
fi
ln -s ${dataset_path}/slim/data demo/data
if [ -d "demo/pretrain" ];then
   rm -rf demo/pretrain
fi
ln -s ${dataset_path}/slim/slim_pretrain demo/pretrain


./.run_ce.sh
