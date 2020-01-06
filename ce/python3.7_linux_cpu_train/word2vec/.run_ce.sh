#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

export NUM_THREADS=1

rm -rf v1_cpu5_b100_lr1dir
export OPENBLAS_NUM_THREADS=1
export CPU_NUM=1
FLAGS_benchmark=true python train.py --enable_ce --train_data_dir data/convert_1-billion-word --dict_path data/test_build_dict --num_passes 1 --batch_size 100 --model_output_dir v1_cpu1_b100_lr1dir --base_lr 1.0 --print_batch 1000 --with_speed --is_sparse >word2vec_trian_all_cpu1.log 2>&1
cat word2vec_trian_all_cpu1.log|grep 31178000 |awk -F ' ' 'NR==1{print "kpis\ttest_cpu1_loss\t"$10}' | python _ce.py

rm -rf v1_cpu5_b100_lr1dir
export OPENBLAS_NUM_THREADS=1
export CPU_NUM=5
FLAGS_benchmark=true python train.py --enable_ce --train_data_dir data/convert_1-billion-word --dict_path data/test_build_dict --num_passes 1 --batch_size 100 --model_output_dir v1_cpu1_b100_lr1dir --base_lr 1.0 --print_batch 1000 --with_speed --is_sparse >word2vec_trian_all_cpu5.log 2>&1
cat word2vec_trian_all_cpu5.log|grep 6235000 |awk -F ' ' 'NR==1{print "kpis\ttest_cpu5_loss\t"$10}' | python _ce.py
