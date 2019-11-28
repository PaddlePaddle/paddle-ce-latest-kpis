#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

export CPU_NUM=1
export NUM_THREADS=1
python train.py --train_dir train_data --use_cuda 0 --batch_size 50 --model_dir model_output --epochs 2 --enable_ce --step_num 1000 | python _ce.py