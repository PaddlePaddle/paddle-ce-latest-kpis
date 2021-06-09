#!/usr/bin/env bash

cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name install!!!"

#路径配置
root_path=$cur_path/../../

log_path=$root_path/log/$model_name/
mkdir -p $log_path

print_info(){
if [ $1 -ne 0 ];then
    cat ${log_path}/$2.log
    echo "exit_code: 1.0" >> ${log_path}/S_$2.log
else
    echo "exit_code: 0.0" >> ${log_path}/S_$2.log
fi
}

hub config server==http://paddlepaddle.org.cn/paddlehub
hub install chinese_text_detection_db_mobile > ${log_path}/$1.log 2>&1
print_info $? install
