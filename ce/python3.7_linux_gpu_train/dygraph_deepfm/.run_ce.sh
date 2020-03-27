#!/bin/bash
# This file is only used for continuous evaluation.

export CUDA_VISIBLE_DEVICES=3
if [ -d 'models' ]; then
    rm -rf models
fi
python train.py --num_epoch 1 >deepfm_dy_gpu1.log 2>&1
cat deepfm_dy_gpu1.log|grep -E "finished|test" |awk -F ' ' 'NR==1{print "kpis\teach_pass_duration_gpu1\t"$9}NR==3{print "kpis\ttest_auc_gpu1\t"$9}' | python _ce.py

