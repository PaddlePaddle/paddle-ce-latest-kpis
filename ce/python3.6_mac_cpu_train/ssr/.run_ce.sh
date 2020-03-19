#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

export CPU_NUM=1
export NUM_THREADS=1
python train.py --train_dir train_data --use_cuda 0 --batch_size 50 --model_dir model_output --epochs 2 --enable_ce --step_num 1000 | python _ce.py
# infer
python infer.py --test_dir test_data --use_cuda 0 --batch_size 50 --model_dir model_output --start_index 1 --last_index 1 > $log_path/ssr_I.log 2>&1
if [ $? -ne 0 ];then↩
        mv ${log_path}/ssr_I.log ${log_path}/FAIL/ssr_I.log↩
        echo -e "ssr,infer,FAIL" >>${log_path}/result.log
else↩
        mv ${log_path}/ssr_I.log ${log_path}/SUCCESS/ssr_I.log↩
        echo -e "ssr,infer,SUCCESS" >>${log_path}/result.log↩
fi
