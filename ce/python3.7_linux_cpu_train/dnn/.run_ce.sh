#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1


#cudaid=${face_detection:=0} # use 0-th card as default
#export CUDA_VISIBLE_DEVICES=$cudaid
export CPU_NUM=1
export NUM_THREADS=1

FLAGS_benchmark=true  python train.py --is_local 1 --cloud_train 0 --train_data_path data/raw/train.txt --enable_ce >log_cpu1
cat log_cpu1 | python _ce.py

export CPU_NUM=1
export NUM_THREADS=8

FLAGS_benchmark=true  python train.py --is_local 1 --cloud_train 0 --train_data_path data/raw/train.txt --enable_ce >log_cpu1threads8
cat log_cpu1threads8 | python _ce.py

export CPU_NUM=8
export NUM_THREADS=8

FLAGS_benchmark=true  python train.py --is_local 1 --cloud_train 0 --train_data_path data/raw/train.txt --enable_ce >log_cpu8thread8
cat log_cpu8thread8 | python _ce.py
