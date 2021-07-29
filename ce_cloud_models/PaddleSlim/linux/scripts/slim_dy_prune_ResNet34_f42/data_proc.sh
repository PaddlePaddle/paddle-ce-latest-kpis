#!/usr/bin/env bash

#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}
echo "$model_name 模型数据预处理阶段"
#配置目标数据存储路径
root_path=$cur_path/../../
code_path=$cur_path/../../PaddleSlim/demo/dygraph/pruning/
#临时环境更改


#获取数据逻辑
cd /home/data/cfs/models_ce/ILSVRC2012_data_demo/ILSVRC2012
ls
cd ${root_path}/PaddleSlim/demo/dygraph/pruning/
pwd
if [ "$1" = "demo" ];then   # 小数据集
    if [ ! -d "data" ];then
        wget -q https://sys-p0.bj.bcebos.com/slim_ci/ILSVRC2012_data_demo.tar.gz --no-check-certificate
        tar xf ILSVRC2012_data_demo.tar.gz
        mv ILSVRC2012_data_demo data
    fi
elif [ "$1" = "all" ];then   # 全量数据集
    if [ ! -d "data" ];then
        mkdir data && cd data;
        ln -s /home/data/cfs/models_ce/ILSVRC2012_data_demo/ILSVRC2012 ILSVRC2012;
        pwd
        cd ILSVRC2012
        pwd
        ls
    fi
fi


# download pretrain model

#数据处理逻辑


