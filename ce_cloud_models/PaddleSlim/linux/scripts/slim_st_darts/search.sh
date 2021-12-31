#!/usr/bin/env bash
#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'recv' 恢复训练
# $2:  $XPU = gpu or cpu
#获取当前路径
cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name 模型train阶段"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../PaddleSlim/demo/darts
log_path=$root_path/log/$model_name/
mkdir -p $log_path
#临时环境更改


#访问RD程序
print_info(){
if [ $1 -ne 0 ];then
    echo "exit_code: 1.0" >> ${log_path}/$2.log
    echo -e "\033[31m FAIL_$2 \033[0m"
    echo $2 fail log as follows
    cat ${log_path}/$2.log
    cp ${log_path}/$2.log ${log_path}/FAIL_$2.log
else
    tail -80 ${log_path}/$2.log |grep Epoch |grep acc_1 |awk -F ' ' \
        '{print"\ntop1:" $11"\ttop5:"$13}' | tr -d ',' >> ${log_path}/$2.log
    echo "exit_code: 0.0" >> ${log_path}/$2.log
fi
}

cd $code_path

echo -e "\033[32m `pwd` train \033[0m";


if [ "$1" = "linux_st_gpu1" ];then #单卡
    python search.py --epochs 1 \
    --use_multiprocess False \
    --batch_size 32 > ${log_path}/$2.log 2>&1
    print_info $? $2
elif [ "$1" = "linux_st_gpu2" ];then #多卡
    python search.py --epochs 1 \
    --use_multiprocess False \
    --batch_size 32 > ${log_path}/$2.log 2>&1
    print_info $? $2
elif [ "$1" = "linux_st_cpu" ];then #CPU
    python search.py --epochs 1 \
    --use_multiprocess False \
    --batch_size 32 \
    --use_gpu False > ${log_path}/$2.log 2>&1
    print_info $? $2
fi
