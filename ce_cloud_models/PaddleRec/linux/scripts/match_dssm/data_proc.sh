#!/usr/bin/env bash

#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name get data"
#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../PaddleRec/datasets

#临时环境更改
if [ "$1" = "down_BQ_dssm" ];then # 下载数据
    cd $code_path/BQ_dssm
    sh run.sh
    cd $data_path
    if [ ! -d 'BQ_dssm' ];then
        cp -r $code_path/BQ_dssm $data_path/
    fi
    cd -
elif [ "$1" = "down_letor07" ];then
    cd $code_path/letor07
    sh run.sh
    cd $data_path
    if [ ! -d 'letor07' ];then
        cp -r $code_path/letor07 $data_path/
    fi
    cd -
elif [ "$1" = "down_BQ_simnet" ];then
    cd $code_path/BQ_simnet
    sh run.sh
    cd $data_path
    if [ ! -d 'BQ_simnet' ];then
        cp -r $code_path/BQ_simnet $data_path/
    fi
    cd -
elif [ "$1" = "local_BQ_dssm" ];then
    # 软连数据
    cd $code_path/;
    rm -rf BQ_dssm;
    ln -s $data_path/BQ_dssm;
    ls;
elif [ "$1" = "local_letor07" ];then
    # 软连数据
    cd $code_path/;
    rm -rf letor07;
    ln -s $data_path/letor07;
    ls;
elif [ "$1" = "local_BQ_simnet" ];then
    # 软连数据
    cd $code_path/;
    rm -rf BQ_simnet;
    ln -s $data_path/BQ_simnet;
    ls;
else
    echo "$model_name data_proc.sh  parameters error "

fi
#获取数据逻辑

#数据处理逻辑


