#!/usr/bin/env bash
#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'cpu' cpu 训练
#获取当前路径
cur_path=`pwd`
model_name=$2
temp_path=wide_deep

echo "$2 train"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../PaddleRec/models/rank/${temp_path}
log_path=$root_path/log/rank_${temp_path}/
mkdir -p $log_path
#临时环境更改

#访问RD程序
print_info(){
if [ $1 -ne 0 ];then
    tail -100 ${log_path}/$2.log
    mv ${log_path}/$2.log ${log_path}/F_$2.log
    echo -e "\033[31m ${log_path}/F_$2 \033[0m"
else
#    cat ${log_path}/$2.log
    mv ${log_path}/$2.log ${log_path}/S_$2.log
    echo -e "\033[32m ${log_path}/S_$2 \033[0m"
fi
}


cd $code_path/
echo -e "\033[32m `pwd` train \033[0m";

sed -i "s/  epochs: 4/  epochs: 1/g" config_bigdata.yaml
sed -i "s/  infer_end_epoch: 4/  infer_end_epoch: 1/g" config_bigdata.yaml

rm -rf output*

if [ "$1" = "linux_dy_gpu1" ];then #单卡
    python -u ../../../tools/trainer.py -m config_bigdata.yaml \
    -o runner.use_gpu=True > ${log_path}/$2.log 2>&1
    print_info $? $2
elif [ "$1" = "linux_dy_gpu2" ];then #多卡
    # 多卡的运行方式
    python -m paddle.distributed.launch ../../../tools/trainer.py -m config_bigdata.yaml \
    -o runner.use_gpu=True runner.use_fleet=True > ${log_path}/$2.log 2>&1
    print_info $? $2
    mv $code_path/log $log_path/$2_dist_log
elif [ "$1" = "linux_dy_cpu" ];then
    python -u ../../../tools/trainer.py -m config_bigdata.yaml \
    -o runner.use_gpu=False > ${log_path}/$2.log 2>&1
    print_info $? $2

elif [ "$1" = "linux_st_gpu1" ];then #单卡
    python -u ../../../tools/static_trainer.py -m config_bigdata.yaml \
    -o runner.use_gpu=True > ${log_path}/$2.log 2>&1
    print_info $? $2
elif [ "$1" = "linux_st_gpu2" ];then #多卡
    # 多卡的运行方式
    python -m paddle.distributed.launch ../../../tools/static_trainer.py -m config_bigdata.yaml \
    -o runner.use_gpu=True runner.use_fleet=True > ${log_path}/$2.log 2>&1
    print_info $? $2
    mv $code_path/log $log_path/$2_dist_log

elif [ "$1" = "linux_st_cpu" ];then
    python -u ../../../tools/static_trainer.py -m config_bigdata.yaml \
    -o runner.use_gpu=False > ${log_path}/$2.log 2>&1
    print_info $? $2
else
    echo "$model_name train.sh  parameters error "
fi