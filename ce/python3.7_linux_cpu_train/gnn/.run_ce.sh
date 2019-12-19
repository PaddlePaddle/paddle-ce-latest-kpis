#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

export CUDA_VISIBLE_DEVICES=0
export CPU_NUM=1
export NUM_THREADS=1

FLAGS_benchmark=true  python -u train.py --use_cuda 0 --epoch_num 5 --enable_ce 1> log
cat log | python _ce.py 



