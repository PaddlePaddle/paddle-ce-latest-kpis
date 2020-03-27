#!/bin/bash
# This file is only used for continuous evaluation.

export CUDA_VISIBLE_DEVICES=1
if [ -d 'models' ]; then
    rm -rf models
fi
python local_train.py --enable_ce --model_output_dir models --num_epoch 1 >xdeepfm_T_gpu1.log 2>&1
cat xdeepfm_T_gpu1.log|grep data: |awk -F "[" 'END{print "kpis\ttest_auc_gpu1\t"$2}' |tr -d ']' | python _ce.py

export CUDA_VISIBLE_DEVICES=1,2
if [ -d 'models' ]; then
    rm -rf models
fi
python local_train.py --enable_ce --model_output_dir models --num_epoch 1 >xdeepfm_T_gpu2.log 2>&1
cat xdeepfm_T_gpu2.log|grep data: |awk -F "[" 'END{print "kpis\ttest_auc_gpu2\t"$2}' |tr -d ']' | python _ce.py

#infer
python infer.py --model_output_dir models --test_epoch 0 >test.log
if [ $? -ne 0 ];then
    echo -e "xdeepfm,infer,FAIL"
else
    echo -e "xdeepfm,infer,SUCCESS"
fi
