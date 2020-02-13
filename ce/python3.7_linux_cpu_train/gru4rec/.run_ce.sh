#!/bin/bash
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

export CPU_NUM=1
export NUM_THREADS=1

FLAGS_benchmark=true  python train.py --train_dir train_big_data --vocab_path vocab_big.txt --use_cuda 0 --batch_size 500 --model_dir model_output --pass_num 2 --enable_ce --step_num 10 >log_cpu
cat log_cpu | python _ce.py

