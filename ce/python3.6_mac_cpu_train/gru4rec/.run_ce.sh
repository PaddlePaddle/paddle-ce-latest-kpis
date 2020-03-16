#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

export CPU_NUM=1
export NUM_THREADS=1
export ce_mode=1
python train.py --train_dir train_data --use_cuda 0 --pass_num 1 --model_dir output --pass_num 2 --enable_ce --step_num 10 | python _ce.py
