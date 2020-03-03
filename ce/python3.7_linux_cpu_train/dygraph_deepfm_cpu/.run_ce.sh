#!/bin/bash
# This file is only used for continuous evaluation.

export CPU_NUM=1
if [ -d 'models' ]; then
    rm -rf models
fi
python train.py --num_epoch 1 --use_gpu False  >deepfm_dy_cpu1.log 2>&1
cat deepfm_dy_cpu1.log|grep -E "finished|test" |awk -F ' ' 'NR==1{print "kpis\teach_pass_duration_cpu1\t"$9}NR==3{print "kpis\ttest_auc_cpu1\t"$9}' | python _ce.py
