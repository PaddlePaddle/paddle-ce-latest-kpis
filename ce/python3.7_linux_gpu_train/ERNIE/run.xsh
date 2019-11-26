#/bin/bash
export ernie_dir=$PWD/../../ernie_repo
#copy models files
rm -rf ${ernie_dir}/.run_ce.sh
cp -r ${ernie_dir}/. ./
if [ -d "config" ];then rm -rf config
fi
if [ -d "ERNIE_1.0.1" ];then rm -rf ERNIE_1.0.1
fi
if [ -d "task_data" ];then rm -rf task_data
fi
ln -s ${dataset_path}/ERNIE/config config
ln -s ${dataset_path}/ERNIE/ERNIE_1.0.1 ERNIE_1.0.1
ln -s ${dataset_path}/ERNIE/task_data task_data

./.run_ce.sh
