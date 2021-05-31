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
if [ "$1" = "down_movielens_pinterest_NCF" ];then # 下载数据
    cd $code_path/movielens_pinterest_NCF
    sh run.sh
    cd $data_path
    if [ ! -d 'movielens_pinterest_NCF' ];then
        cp -r $code_path/movielens_pinterest_NCF $data_path/
    fi
    cd -
elif [ "$1" = "down_senti_clas" ];then
    cd $code_path/senti_clas
    sh run.sh
    cd $data_path
    if [ ! -d 'senti_clas' ];then
        cp -r $code_path/senti_clas $data_path/
    fi
    cd -
elif [ "$1" = "local_movielens_pinterest_NCF" ];then
    # 软连数据
    cd $code_path/
    rm -rf movielens_pinterest_NCF;
    ln -s $data_path/movielens_pinterest_NCF;
    ls;
elif [ "$1" = "local_senti_clas" ];then
    # 软连数据
    cd $code_path/
    rm -rf senti_clas;
    ln -s $data_path/senti_clas;
    ls;
else
    echo "$model_name data_proc.sh  parameters error "

fi
#获取数据逻辑

#数据处理逻辑


