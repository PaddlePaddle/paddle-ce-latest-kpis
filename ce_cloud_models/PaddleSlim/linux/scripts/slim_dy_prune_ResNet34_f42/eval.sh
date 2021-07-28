#!/usr/bin/env bash
#外部传入参数说明
# $1:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型eval阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../PaddleSlim/demo/dygraph/pruning/
log_path=$root_path/log/$model_name/
mkdir -p $log_path
#临时环境更改

#访问RD程序
print_info(){
if [ $1 -ne 0 ];then
    cat ${log_path}/$2.log
    echo -e "\033[31m ${log_path}/F_$2 \033[0m"
    echo "exit_code: 1.0" >>${log_path}/$2.log
else
    echo -e "\033[32m ${log_path}/S_$2 \033[0m"
    echo "exit_code: 0.0" >>${log_path}/$2.log
fi
}

cd $code_path

if [ "$1" = "linux_dy_gpu1" ];then #单卡
    python eval.py \
    --checkpoint=./fpgm_resnet34_025_120_models/1 \
    --model="resnet34" \
    --pruned_ratio=0.25 \
    --batch_size=128 > ${log_path}/$2.log 2>&1
    print_info $? $2
elif [ "$1" = "linux_dy_gpu2" ];then # 多卡
    python -m paddle.distributed.launch eval.py \
    --checkpoint=./fpgm_resnet34_025_120_models/1 \
    --model="resnet34" \
    --pruned_ratio=0.25 \
    --batch_size=128 > ${log_path}/$2.log 2>&1
    print_info $? $2
elif [ "$1" = "linux_dy_cpu" ];then # cpu
    python eval.py \
    --use_gpu False \
    --checkpoint=./fpgm_resnet34_025_120_models/1 \
    --model="resnet34" \
    --pruned_ratio=0.25 \
    --batch_size=128 > ${log_path}/$2.log 2>&1
    print_info $? $2
fi
