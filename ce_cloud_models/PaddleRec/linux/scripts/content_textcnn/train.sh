#!/usr/bin/env bash
#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'cpu' cpu 训练
#获取当前路径
cur_path=`pwd`
model_name=$2
temp_path=$(echo $2|awk -F '_' '{print $2}')

echo "$2 train"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../PaddleRec/models/contentunderstanding/${temp_path}/
log_path=$root_path/log/content_${temp_path}/
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
    echo -e "\033[32m ${log_path}/S_$2 \033[0m"
fi
}


cd $code_path
echo -e "\033[32m `pwd` train \033[0m";
# 数据集太小，直接就跑收敛性的

case $1 in
 linux_dy_gpu1)
        echo "start.$2.$1."
        rm -rf output
        sed -i "s/  use_gpu: False/  use_gpu: True/g" config_bigdata.yaml
        python -u ../../../tools/trainer.py -m config_bigdata.yaml > ${log_path}/$2_T.log 2>&1
        print_info $? $2_T
        python -u ../../../tools/infer.py -m config_bigdata.yaml > ${log_path}/$2_I.log 2>&1
        print_info $? $2_I
        ;;
 linux_dy_gpu2)
        echo "start.$2.$1."
        rm -rf output
        sed -i "s/  use_gpu: False/  use_gpu: True/g" config_bigdata.yaml
        sed -i '/runner:/a\  use_fleet: True' config.yaml
        # 多卡的运行方式
        fleetrun ../../../tools/trainer.py -m config_bigdata.yaml > ${log_path}/$2_T.log 2>&1
        print_info $? $2_T
        mv $code_path/log $log_path/$2_dist_train_log
        fleetrun ../../../tools/infer.py -m config_bigdata.yaml > ${log_path}/$2_I.log 2>&1
        print_info $? $2_I
        mv $code_path/log $log_path/$2_dist_infer_log
        ;;
 linux_dy_cpu)
        echo "start.$2.$1."
        rm -rf output
        sed -i "s/  use_gpu: True/  use_gpu: False/g" config_bigdata.yaml
        python -u ../../../tools/trainer.py -m config_bigdata.yaml > ${log_path}/$2_T.log 2>&1
        print_info $? $2_T
        python -u ../../../tools/infer.py -m config_bigdata.yaml > ${log_path}/$2_I.log 2>&1
        print_info $? $2_I
        ;;
 linux_st_gpu1)
        echo "start.$2.$1."
        rm -rf output
        sed -i "s/  use_gpu: False/  use_gpu: True/g" config_bigdata.yaml
        python -u ../../../tools/static_trainer.py -m config_bigdata.yaml > ${log_path}/$2_T.log 2>&1
        print_info $? $2_T
        python -u ../../../tools/static_infer.py -m config_bigdata.yaml > ${log_path}/$2_I.log 2>&1
        print_info $? $2_I
        ;;
 linux_st_gpu2)
        echo "start.$2.$1."
        rm -rf output
        sed -i "s/  use_gpu: False/  use_gpu: True/g" config_bigdata.yaml
        sed -i '/runner:/a\  use_fleet: True' config.yaml
        # 多卡的运行方式
        fleetrun ../../../tools/static_trainer.py -m config_bigdata.yaml > ${log_path}/$2_T.log 2>&1
        print_info $? $2_T
        mv $code_path/log $log_path/$2_dist_train_log
        fleetrun ../../../tools/static_infer.py -m config_bigdata.yaml > ${log_path}/$2_I.log 2>&1
        print_info $? $2_I
        mv $code_path/log $log_path/$2_dist_infer_log
        ;;
 linux_st_cpu)
        echo "start.$2.$1."
        rm -rf output
        sed -i "s/  use_gpu: True/  use_gpu: False/g" config_bigdata.yaml
        python -u ../../../tools/static_trainer.py -m config_bigdata.yaml > ${log_path}/$2_T.log 2>&1
        print_info $? $2_T
        python -u ../../../tools/static_infer.py -m config_bigdata.yaml > ${log_path}/$2_I.log 2>&1
        print_info $? $2_I
        ;;
 *)
        echo "Usage: $1 [linux_dy_gpu1|linux_dy_gpu1_con]"
        ;;
esac
