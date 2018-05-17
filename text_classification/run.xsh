#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
cudaid=${text_classification:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

#LSTM pass_num 15
FLAGS_benchmark=true  python train.py lstm
