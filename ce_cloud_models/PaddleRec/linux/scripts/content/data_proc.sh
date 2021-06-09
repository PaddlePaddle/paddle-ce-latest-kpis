#!/usr/bin/env bash

#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name get data"
#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../PaddleRec/datasets

#临时环境更改
if [ "$1" = "down_ag_news" ];then # 下载数据
    cd $code_path/ag_news
    sh run.sh
elif [ "$1" = "down_senti_clas" ];then
    cd $code_path/senti_clas
    sh run.sh
else
    echo "$model_name data_proc.sh  parameters error "

fi
#获取数据逻辑

#数据处理逻辑


