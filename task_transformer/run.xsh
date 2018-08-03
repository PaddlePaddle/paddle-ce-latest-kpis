#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
cudaid=${transformer_cudaid:=0}
export CUDA_VISIBLE_DEVICES=$cudaid
FLAGS_benchmark=true python train.py --gpu_card_num 1


cudaid=${transformer_cudaid_m:=0,1,2,3} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid
FLAGS_benchmark=true python train.py --gpu_card_num 4
