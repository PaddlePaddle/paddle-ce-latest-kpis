#!/usr/bin/env bash

#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name get data"
#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../PaddleRec/datasets

data_path=/ssd2/ce_data/rec_datasets
#临时环境更改
if [ "$1" = "down_criteo" ];then # 下载数据
    cd $code_path/criteo
    sh run.sh
    cd $data_path
    if [ ! -d 'criteo' ];then
        cp -r $code_path/criteo $data_path/
    fi
    cd -
elif [ "$1" = "down_lr" ];then
    cd $code_path/criteo_lr
    sh run.sh
    cd $data_path
    if [ ! -d 'criteo_lr' ];then
        cp -r $code_path/criteo_lr $data_path/
    fi
    cd -
elif [ "$1" = "local_criteo" ];then
    # 软连数据
    cd $code_path/criteo
    if [ ! -d 'slot_train_data_full' ];then
        ln -s $data_path/criteo/slot_train_data_full;
        ln -s $data_path/criteo/slot_test_data_full;
    fi
    ls;
elif [ "$1" = "local_lr" ];then
    # 软连数据
    cd $code_path/criteo_lr
    if [ ! -d 'slot_test_data' ];then
        ln -s $data_path/criteo_lr/slot_test_data;
        ln -s $data_path/criteo_lr/slot_train_data;
    fi
    ls;
else
    echo "$model_name data_proc.sh  parameters error "

fi
#获取数据逻辑

#数据处理逻辑


