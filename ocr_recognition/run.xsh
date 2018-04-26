#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
cudaid=${ocr_recognition_cudaid:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true FLAGS_fraction_of_gpu_memory_to_use=0.9 python ctc_train.py --use_gpu=True --batch_size=128 --pass_num=1  --iterations=10000
