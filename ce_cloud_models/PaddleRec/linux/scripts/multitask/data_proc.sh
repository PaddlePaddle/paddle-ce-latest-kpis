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
if [ "$1" = "down_ali-ccp" ];then # 下载数据
    cd $code_path/ali-ccp
    sh run.sh
    cd $data_path
    if [ ! -d 'ali-ccp' ];then
        cp -r $code_path/ali-ccp $data_path/
    fi
    cd -
elif [ "$1" = "down_census" ];then
    cd $code_path/census
    sh run.sh
    cd $data_path
    if [ ! -d 'census' ];then
        cp -r $code_path/census $data_path/
    fi
    cd -
elif [ "$1" = "local_ali-ccp" ];then
    # 软连数据
    cd $code_path/
    ln -s $data_path/ali-ccp;
    ls;
elif [ "$1" = "local_census" ];then
    # 软连数据
    cd $code_path/
    ln -s $data_path/census;
    ls;
else
    echo "$model_name data_proc.sh  parameters error "

fi
#获取数据逻辑

#数据处理逻辑


