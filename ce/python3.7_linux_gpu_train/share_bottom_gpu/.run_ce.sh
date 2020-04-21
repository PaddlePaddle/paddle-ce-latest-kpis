#!/bin/bash
# share_bottom gpu1
model=share_bottom_gpu1
CUDA_VISIBLE_DEVICES=6 python share_bottom.py  \
                        --use_gpu 1\
                        --train_data_path train_data/\
                        --test_data_path test_data/\
                        --batch_size 32\
                        --feature_size 499\
                        --bottom_size 117\
                        --tower_nums 2\
                        --tower_size 8\
                        --model_dir model_dir\
                        --epochs 2 >${model} 2>&1
cat ${model}|grep epoch_time|awk -F ' ' 'END{print "kpis\t""'${model}_epoch_time'""\t"$7}' >>${model}_ce
cat ${model}|grep mean_sb_test_auc_income|awk -F ' |,' 'END{print "kpis\t""'${model}_mean_sb_test_auc_income'""\t"$7"\nkpis\t""'${model}_mean_sb_test_auc_marital'""\t"$9"\nkpis\t""'${model}_max_sb_test_auc_income'""\t"$11"\nkpis\t""'${model}_max_sb_test_auc_marital'""\t"$13}' >>${model}_ce
cat ${model}_ce |python _ce.py
