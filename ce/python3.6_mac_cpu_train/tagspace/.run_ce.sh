#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1


export CPU_NUM=1
export NUM_THREADS=1
python train.py  --use_cuda 0 --enable_ce | python _ce.py
# infer
python infer.py --use_cuda 0 > $log_path/tagsapce_I.log 2>&1
if [ $? -ne 0 ];then↩
        mv ${log_path}/tagsapce_I.log ${log_path}/FAIL/tagsapce_I.log↩
        echo -e "tagsapce,infer,FAIL" >>${log_path}/result.log
else↩
        mv ${log_path}/tagsapce_I.log ${log_path}/SUCCESS/tagsapce_I.log↩
        echo -e "tagsapce,infer,SUCCESS" >>${log_path}/result.log↩
fi
