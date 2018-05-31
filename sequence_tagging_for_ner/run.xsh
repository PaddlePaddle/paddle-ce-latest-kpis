#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

cudaid=${sequence_tagging:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid
#pass_num 2200
sh download.sh
FLAGS_benchmark=true  python train.py

cudaid=${sequence_tagging_m:=0,1,2,3} # use multi card as default
export CUDA_VISIBLE_DEVICES=$cudaid
#pass_num 2200
sh download.sh
FLAGS_benchmark=true  python train.py --gpu_card_num 4
