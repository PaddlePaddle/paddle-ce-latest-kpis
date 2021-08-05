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
code_path=$cur_path/../../PaddleSlim/demo/distillation
log_path=$root_path/log/$model_name/
mkdir -p $log_path
#临时环境更改


#访问RD程序
print_info(){
if [ $1 -ne 0 ];then
    cat ${log_path}/$2.log
    mv ${log_path}/$2.log ${log_path}/F_$2.log
    echo -e "\033[31m ${log_path}/F_$2 \033[0m"
else
#    cat ${log_path}/$2.log
    mv ${log_path}/$2.log ${log_path}/S_$2.log
    tail -1 ${log_path}/S_$2.log|grep epoch |grep top1 |awk -F ' ' '{print"top1:" $6"\ttop5:"$8}' | tr -d ',' >>${log_path}/S_$2.log
    echo -e "\033[32m ${log_path}/S_$2 \033[0m"
fi
}

cd $code_path
echo -e "\033[32m `pwd` train \033[0m";

if [ "$1" = "linux_st_gpu1" ];then #单卡
    python distill.py \
    --num_epochs 5 \
    --batch_size 512 \
    --save_inference True \
    --model MobileNetV2_x0_25 \
    --teacher_model MobileNetV2 \
    --teacher_pretrained_model ../pretrain/ResNet101_vd_pretrained > ${log_path}/$2.log 2>&1
    print_info $? $2
 #   cat ${log_path}/S_$2.log|grep epoch |grep top1 |tr -d ','|awk -F ' ' 'END{print "acc_top1:\t"$9"\tacctop5:\t"$11}' >>S_$2.log

elif [ "$1" = "linux_st_gpu2" ];then #多卡
    python distill.py \
    --num_epochs 5 \
    --batch_size 512 \
    --save_inference True \
    --model MobileNetV2_x0_25 \
    --teacher_model MobileNetV2 \
    --teacher_pretrained_model ../pretrain/ResNet101_vd_pretrained > ${log_path}/$2.log 2>&1
    print_info $? $2
#    cat ${log_path}/S_$2.log|grep epoch |grep top1 |tr -d ','|awk -F ' ' 'END{print "acc_top1:\t"$9"\tacctop5:\t"$11}' >>S_$2.log

elif [ "$1" = "linux_st_cpu" ];then #CPU
    python distill.py \
    --num_epochs 5 \
    --batch_size 512 \
    --save_inference True \
    --model MobileNetV2_x0_25 \
    --teacher_model MobileNetV2 \
    --teacher_pretrained_model ../pretrain/ResNet101_vd_pretrained \
    --use_gpu False > ${log_path}/$2.log 2>&1
    print_info $? $2
#    cat ${log_path}/S_$2.log|grep epoch |grep top1 |tr -d ','|awk -F ' ' 'END{print "acc_top1:\t"$9"\tacctop5:\t"$11}' >>S_$2.log

fi
