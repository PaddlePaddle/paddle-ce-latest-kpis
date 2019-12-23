#!/bin/bash
# This file is only used for continuous evaluation.

export CPU_NUM=1
if [ -d 'models' ]; then
    rm -rf models
fi
python local_train.py --enable_ce  --num_thread 1 --steps 1000 >dcn_cpu1_thread1.log 2>&1
cat dcn_cpu1_thread1.log|grep data: |awk -F "[" 'END{print "kpis\ttest_auc_cpu1_thread1\t"$2}' |tr -d ']' | python _ce.py

export CPU_NUM=1
if [ -d 'models' ]; then
    rm -rf models
fi
python local_train.py --enable_ce  --num_thread 20 --steps 1000 >dcn_cpu1_thread20.log 2>&1
cat dcn_cpu1_thread20.log|grep data: |awk -F "[" 'END{print "kpis\ttest_auc_cpu1_thread20\t"$2}' |tr -d ']' | python _ce.py

export CPU_NUM=5
if [ -d 'models' ]; then
    rm -rf models
fi
python local_train.py --enable_ce  --num_thread 20 --steps 1000 >dcn_cpu5_thread20.log 2>&1
cat dcn_cpu5_thread20.log|grep data: |awk -F "[" 'END{print "kpis\ttest_auc_cpu5_thread20\t"$2}' |tr -d ']' | python _ce.py



