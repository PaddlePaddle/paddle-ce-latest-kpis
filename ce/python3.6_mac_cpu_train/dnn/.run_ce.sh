#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

export CPU_NUM=1
export NUM_THREADS=1

python train.py --is_local 1 --cloud_train 0 --train_data_path data/raw/train.txt --enable_ce | python _ce.py

