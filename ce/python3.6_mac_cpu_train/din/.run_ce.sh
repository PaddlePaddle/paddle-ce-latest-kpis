#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

export CPU_NUM=1
export NUM_THREADS=1
python -u train.py --config_path data/config.txt --train_dir data/paddle_train.txt --batch_size 32 --epoch_num 1 --use_cuda 0 --enable_ce --batch_num 10000 |  python _ce.py
# infer
python infer.py --model_path din_amazon/global_step_50000 --test_path data/paddle_test.txt --use_cuda 0 >$log_path/din_I.log 2>&1
if [ $? -ne 0 ]; then
        mv ${log_path}/din_I.log ${log_path}/FAIL/din_I.log
		echo -e "\033[33m infer of din failed! \033[0m"
        echo -e "din,infer,FAIL" >>${log_path}/result.log
else
        mv ${log_path}/din_I.log ${log_path}/SUCCESS/din_I.log
		echo -e "\033[33m infer of din successfully! \033[0m"
        echo -e "din,infer,SUCCESS" >>${log_path}/result.log
fi

