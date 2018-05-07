#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
cudaid=${sequence_tagging:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

#pass_num 2200
sh download.sh
FLAGS_benchmark=true  python train.py
