#/bin/bash
export models_dir=$PWD/../../models_repo
#copy models files
rm -rf ${models_dir}/PaddleSlim/.run_ce.sh
cp -r ${models_dir}/PaddleSlim/. ./
if [ -d "data" ];then rm -rf data
fi
if [ -d "pretrain" ];then rm -rf pretrain
fi
ln -s ${dataset_path}/slim/data data
ln -s ${dataset_path}/slim/pretrain pretrain

./.run_ce.sh
