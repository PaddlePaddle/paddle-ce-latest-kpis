#!/usr/bin/env bash

cur_path=`pwd`
model_name=${PWD##*/}

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

if [ $1 == 'infer' ]; then
  if [ $2 == 'single_gpu' ]; then
     echo "$model_name single_gpu export models"
     python point_infer.py > ${log_path}/$3.log 2>&1
     print_info $? $3
  elif [ $2 == 'cpu' ]; then
     echo "$model_name cpu export models"
     python point_infer.py > ${log_path}/$3.log 2>&1
     print_info $? $3
  fi
elif [ $1 == 'onnx' ]; then
  if [ $2 == 'single_gpu' ]; then
     echo "$model_name single_gpu onnx models"
     python point_onnx.py > ${log_path}/$3.log 2>&1
     print_info $? $3
  elif [ $2 == 'cpu' ]; then
     echo "$model_name cpu onnx models"
     python point_onnx.py > ${log_path}/$3.log 2>&1
     print_info $? $3
  fi
fi
