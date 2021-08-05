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
code_path=$cur_path/../../PaddleSlim/demo/quant/quant_embedding
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
    cat ${log_path}/$2.log
    mv ${log_path}/$2.log ${log_path}/S_$2.log
    echo -e "\033[32m ${log_path}/S_$2 \033[0m"
fi
}

cd $code_path

echo -e "\033[32m `pwd` infer \033[0m";

if [ "$1" = "infer1" ];then #单卡
     python infer.py --infer_epoch \
     --test_dir data/test_mid_dir \
     --dict_path data/test_build_dict_word_to_id_ \
     --batch_size 20000 --model_dir v1_cpu5_b100_lr1dir/ \
     --start_index 0 --last_index 0  > ${log_path}/$2.log 2>&1
    print_info $? $2
elif [ "$1" = "infer2" ];then
    python infer.py --infer_epoch \
    --test_dir data/test_mid_dir \
    --dict_path data/test_build_dict_word_to_id_ \
    --batch_size 20000 --model_dir v1_cpu5_b100_lr1dir/  \
    --start_index 0 --last_index 0 --emb_quant True> ${log_path}/$2.log 2>&1
     print_info $? $2
fi
