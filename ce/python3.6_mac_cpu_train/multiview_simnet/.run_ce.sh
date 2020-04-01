#!/bin/bash
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

export CPU_NUM=1
export NUM_THREADS=1
python train.py --enable_ce --epochs=1 | python _ce.py
# infer
python infer.py > $log_path/multiview_I.log 2>&1
if [ $? -ne 0 ]; then
        mv ${log_path}/multiview_I.log ${log_path}/FAIL/multiview_I.log
		echo -e "\033[33m infer of multiview failed! \033[0m"
        echo -e "multiview,infer,FAIL" >>${log_path}/result.log
else
        mv ${log_path}/multiview_I.log ${log_path}/SUCCESS/multiview_I.log
		echo -e "\033[33m infer of multiview successfully! \033[0m"
        echo -e "multiview,infer,SUCCESS" >>${log_path}/result.log
fi


