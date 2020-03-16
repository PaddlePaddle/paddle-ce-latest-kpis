#!/bin/bash

# cpu1
if [ -d 'models' ]; then
    rm -rf models
fi
python -u gru4rec_dy.py  --data_path data/ --model_type gru4rec --use_gpu False >gru4rec_dy_cpu1
cat gru4rec_dy_cpu1 |grep -E "recall@20|cost" |awk -F ' ' 'NR==1{print "kpis\teach_pass_duration_cpu1\t"$3}END{print "kpis\ttest_auc_cpu1\t"$2}'| python _ce.py
