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
code_path=$cur_path/../../PaddleSlim/demo/deep_mutual_learning
log_path=$root_path/log/$model_name/
mkdir -p $log_path
#临时环境更改


#访问RD程序
print_info(){
if [ $1 -ne 0 ];then
    mv ${log_path}/$2.log ${log_path}/F_$2.log
    echo -e "\033[31m ${log_path}/F_$2 \033[0m"
else
    mv ${log_path}/$2.log ${log_path}/S_$2.log
    cat ${log_path}/S_$2.log| grep best_valid_acc |awk \
        -F ' ' '{print"\nbest_valid_acc:" $11}' >> ${log_path}/S_$2.log
    echo -e "\033[32m ${log_path}/S_$2 \033[0m"
fi
}

cd $code_path
echo -e "\033[32m `pwd` train \033[0m";

if [ "$1" = "linux_st_gpu1" ];then #单卡
    python dml_train.py --models='mobilenet-mobilenet' --epochs 5 --batch_size 64 > ${log_path}/$2.log 2>&1
    print_info $? $2

elif [ "$1" = "linux_st_gpu2" ];then #单卡
    python dml_train.py --models='mobilenet-mobilenet' --epochs 5 --batch_size 64 > ${log_path}/$2.log 2>&1
    print_info $? $2
#    cat ${log_path}/S_$2.log|grep best_valid_acc |awk -F ' ' 'END{print "best_valid_acc:\t"$11}' >>S_$2.log

elif [ "$1" = "linux_st_cpu" ];then #单卡
    python dml_train.py --models='mobilenet-mobilenet' --epochs 5 --batch_size 64 --use_gpu False > ${log_path}/$2.log 2>&1
    print_info $? $2
#    cat ${log_path}/S_$2.log|grep best_valid_acc |awk -F ' ' 'END{print "best_valid_acc:\t"$11}' >>S_$2.log

fi
