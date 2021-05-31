#!/usr/bin/env bash
#外部传入参数说明
# $1: 'single' 单卡训练； 'multi' 多卡训练； 'cpu' cpu 训练
#获取当前路径
cur_path=`pwd`
model_name=$2
temp_path=$(echo $2|awk -F '_' '{print $2}')
echo "$2 infer"

#路径配置
root_path=$cur_path/../../
code_path=$cur_path/../../PaddleRec/models/recall/${temp_path}/
log_path=$root_path/log/recall/
mkdir -p $log_path
#临时环境更改

#访问RD程序,包含eval过程
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

echo "................run................."
python -u ../../../tools/trainer.py -m config_bigdata.yaml &> log_train.txt
python -u ../../../tools/infer.py -m config_bigdata.yaml &> result.txt
python3 evaluate.py

cd $code_path
echo -e "\033[32m `pwd` infer \033[0m";

sed -i "s/  epochs: 2/  epochs: 1/g" config_bigdata.yaml
sed -i "s/  infer_end_epoch: 2/  infer_end_epoch: 1/g" config_bigdata.yaml

if [ "$1" = "linux_dy_gpu1" ];then #单卡
    sed -i "s/  use_gpu: False/  use_gpu: True/g" config_bigdata.yaml
    python -u ../../../tools/infer.py -m config_bigdata.yaml > ${log_path}/$2.log 2>&1
    print_info $? $2
    rm -rf result.txt;
    cp ${log_path}/S_$2.log result.txt

elif [ "$1" = "linux_dy_gpu2" ];then #多卡
    sed -i "s/  use_gpu: False/  use_gpu: True/g" config_bigdata.yaml
    # 多卡的运行方式
    python -m paddle.distributed.launch ../../../tools/infer.py --gpus "0,1" -m config_bigdata.yaml ${log_path}/$2.log 2>&1
    print_info $? $2
    rm -rf result.txt;
    cp log/wokerlog.0 result.txt;

elif [ "$1" = "linux_dy_cpu" ];then
    sed -i "s/  use_gpu: True/  use_gpu: False/g" config_bigdata.yaml
    python -u ../../../tools/infer.py -m config_bigdata.yaml > ${log_path}/$2.log 2>&1
    print_info $? $2
    rm -rf result.txt;
    cp ${log_path}/S_$2.log result.txt

elif [ "$1" = "linux_st_gpu1" ];then #单卡
    sed -i "s/  use_gpu: False/  use_gpu: True/g" config_bigdata.yaml
    python -u ../../../tools/static_infer.py -m config_bigdata.yaml > ${log_path}/$2.log 2>&1
    print_info $? $2
    rm -rf result.txt;
    cp ${log_path}/S_$2.log result.txt

elif [ "$1" = "linux_st_gpu2" ];then #多卡
    sed -i "s/  use_gpu: False/  use_gpu: True/g" config_bigdata.yaml
    # 多卡的运行方式
    python -m paddle.distributed.launch ../../../tools/static_infer.py --gpus "0,1" -m config_bigdata.yaml ${log_path}/$2.log 2>&1
    print_info $? $2
    rm -rf result.txt;
    cp log/wokerlog.0 result.txt;
    mv $code_path/log $log_path/$2_dist_log

elif [ "$1" = "linux_st_cpu" ];then
    sed -i "s/  use_gpu: True/  use_gpu: False/g" config_bigdata.yaml
    python -u ../../../tools/static_infer.py -m config_bigdata.yaml > ${log_path}/$2.log 2>&1
    print_info $? $2
    rm -rf result.txt;
    cp ${log_path}/S_$2.log result.txt
else
    echo "$model_name train.sh  parameter error "
fi

python evaluate.py > ${log_path}/$2_evaluate.log 2>&1
print_info $? $2_evaluate
