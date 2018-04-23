#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
cudaid=${seq2seq_cudaid:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

#imdb 128
FLAGS_benchmark=true FLAGS_fraction_of_gpu_memory_to_use=0.0 python model.py --device=GPU --batch_size=128 --iterations=50 --gpu_id=$cudaid
