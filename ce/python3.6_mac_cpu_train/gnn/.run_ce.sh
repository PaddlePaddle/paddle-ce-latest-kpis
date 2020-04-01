#!/bin/bash
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

export CPU_NUM=1
export NUM_THREADS=1
python -u train.py --use_cuda 0 --epoch_num 1 --enable_ce | python _ce.py 
# infer
python infer.py --last_index 1 --use_cuda 0 >$log_path/gnn_I.log 2>&1
if [ $? -ne 0 ]; then
        mv ${log_path}/gnn_I.log ${log_path}/FAIL/gnn_I.log
		echo -e "\033[33m infer of gnn failed! \033[0m"
        echo -e "gnn,infer,FAIL" >>${log_path}/result.log
else
        mv ${log_path}/gnn_I.log ${log_path}/SUCCESS/gnn_I.log
		echo -e "\033[33m infer of gnn successfully! \033[0m"
        echo -e "gnn,infer,SUCCESS" >>${log_path}/result.log
fi


