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
code_path=$cur_path/../../PaddleSlim/demo/dygraph/quant/
log_path=$root_path/log/$model_name/
mkdir -p $log_path
#临时环境更改


#访问RD程序
print_info(){
if [ $1 -ne 0 ];then
    echo -e "\033[31m ${log_path}/F_$2 \033[0m"
    echo "exit_code: 1.0" >>${log_path}/$2.log
else
    echo -e "\033[32m ${log_path}/S_$2 \033[0m"
    echo "exit_code: 0.0" >>${log_path}/$2.log
fi
echo $2 log as below
cat ${log_path}/$2.log
}

cd $code_path

if [ "$1" = "linux_dy_gpu1" ];then #单卡
    python train.py --model='mobilenet_v1' \
    --pretrained_model '../../pretrain/MobileNetV1_pretrained' \
    --num_epochs 1 \
    --batch_size 128 > ${log_path}/$2.log 2>&1
    print_info $? $2
elif [ "$1" = "linux_dy_gpu2" ];then # 多卡
    python -m paddle.distributed.launch  \
    --log_dir="quant_v1_linux_dy_gpu2_dist_log" train.py \
    --lr=0.001 \
    --pretrained_model '../../pretrain/MobileNetV1_pretrained' \
    --use_pact=True --num_epochs=1 \
    --l2_decay=2e-5 \
    --ls_epsilon=0.1 \
    --batch_size=128 \
    --model_save_dir output > ${log_path}/$2.log 2>&1
    print_info $? $2
    mv $code_path/quant_v1_linux_dy_gpu2_dist_log $log_path/$2_dist_log

elif [ "$1" = "linux_dy_con_gpu2" ];then # 多卡
    python -m paddle.distributed.launch  \
    --log_dir="quant_v1_linux_dy_gpu2_dist_log" train.py \
    --lr=0.001 \
    --pretrained_model '../../pretrain/MobileNetV1_pretrained' \
    --use_pact=True --num_epochs=1 \
    --l2_decay=2e-5 \
    --ls_epsilon=0.1 \
    --batch_size=128 \
    --model_save_dir output > ${log_path}/$2.log 2>&1
    print_info $? $2
    mv $code_path/quant_v1_linux_dy_gpu2_dist_log $log_path/$2_dist_log

elif [ "$1" = "linux_dy_cpu" ];then # cpu
    python train.py  --lr=0.001 \
    --use_gpu False \
    --pretrained_model '../../pretrain/MobileNetV1_pretrained' \
    --use_pact=True --num_epochs=1 \
    --l2_decay=2e-5 \
    --ls_epsilon=0.1 \
    --batch_size=128 \
    --model_save_dir output > ${log_path}/$2.log 2>&1
    print_info $? $2
fi
