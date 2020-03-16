#!/bin/bash

# gpu1
rm -rf models
export CUDA_VISIBLE_DEVICES=3

python -u gru4rec_dy.py  --data_path data/ --model_type gru4rec >gru4rec_dy_gpu1
cat gru4rec_dy_gpu1 |grep -E "recall@20|cost" |awk -F ' ' 'NR==1{print "kpis\teach_pass_duration_gpu1\t"$3}END{print "kpis\ttest_auc_gpu1\t"$2}'| python _ce.py

