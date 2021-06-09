#!/usr/bin/env bash

cur_path=`pwd`
model_name=${PWD##*/}

echo "$model_name predict!!!"

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

if [ $1 == 'seg' ]; then
  if [ $2 == 'single_gpu' ]; then
     python seg_predict_gpu.py > ${log_path}/$3.log 2>&1
     print_info $? $3
  elif [ $2 == 'cpu' ]; then
     python seg_predict_cpu.py > ${log_path}/$3.log 2>&1
     print_info $? $3
fi
fi

if [ $1 == 'video' ]; then
  if [ $2 == 'single_gpu' ]; then
     python seg_video_gpu.py > ${log_path}/$3.log 2>&1
     print_info $? $3
  elif [ $2 == 'cpu' ]; then
     python seg_video_cpu.py > ${log_path}/$3.log 2>&1
     print_info $? $3
fi
fi

if [ $1 == 'video_stream' ]; then
  if [ $2 == 'single_gpu' ]; then
     python seg_video_stream_gpu.py > ${log_path}/$3.log 2>&1
     print_info $? $3
  elif [ $2 == 'cpu' ]; then
     python seg_video_stream_cpu.py > ${log_path}/$3.log 2>&1
     print_info $? $3
fi
fi
