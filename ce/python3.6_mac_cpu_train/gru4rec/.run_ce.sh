#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

export CPU_NUM=1
export NUM_THREADS=1
export ce_mode=1
python train.py --train_dir train_data --use_cuda 0 --pass_num 1 --model_dir output --pass_num 2 --enable_ce --step_num 10 | python _ce.py

python train_sample_neg.py --loss bpr --use_cuda 0 --pass_num 1 > $log_path/gru4rec_neg_bpr_T.log 2>&1
if [ $? -ne 0 ]; then
        mv ${log_path}/gru4rec_neg_bpr_T.log ${log_path}/FAIL/gru4rec_neg_bpr_T.log
		echo -e "\033[33m infer of gru4rec_neg_bpr failed! \033[0m"
        echo -e "gru4rec_neg_bpr,train,FAIL" >>${log_path}/result.log
else
        mv ${log_path}/gru4rec_neg_bpr_T.log ${log_path}/SUCCESS/gru4rec_neg_bpr_T.log
		echo -e "\033[33m infer of gru4rec_neg_bpr successfully! \033[0m"
        echo -e "gru4rec_neg_bpr,train,SUCCESS" >>${log_path}/result.log
fi
python train_sample_neg.py --loss ce --use_cuda 0 --pass_num 1 > $log_path/gru4rec_neg_ce_T.log 2>&1
if [ $? -ne 0 ]; then
        mv ${log_path}/gru4rec_neg_ce_T.log ${log_path}/FAIL/gru4rec_neg_ce_T.log
		echo -e "\033[33m infer of gru4rec_neg_ce failed! \033[0m"
        echo -e "gru4rec_neg_ce,train,FAIL" >>${log_path}/result.log
else
        mv ${log_path}/gru4rec_neg_ce_T.log ${log_path}/SUCCESS/gru4rec_neg_ce_T.log
		echo -e "\033[33m infer of gru4rec_neg_ce successfully! \033[0m"
        echo -e "gru4rec_neg_ce,train,SUCCESS" >>${log_path}/result.log
fi
python infer.py --test_dir test_data/ --model_dir output/ --start_index 1 --last_index 1 --use_cuda 0 > %log_path%/gru4rec_I.log 2>&1
if [ $? -ne 0 ]; then
        mv ${log_path}/gru4rec_I.log ${log_path}/FAIL/gru4rec_I.log
		echo -e "\033[33m infer of gru4rec failed! \033[0m"
        echo -e "gru4rec,infer,FAIL" >>${log_path}/result.log
else
        mv ${log_path}/gru4rec_I.log ${log_path}/SUCCESS/gru4rec_I.log
		echo -e "\033[33m infer of gru4rec successfully! \033[0m"
        echo -e "gru4rec,infer,SUCCESS" >>${log_path}/result.log
fi
