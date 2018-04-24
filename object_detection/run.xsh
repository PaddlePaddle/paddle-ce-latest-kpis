#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
cudaid=${object_detection_cudaid:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

if [ ! -d "data/pascalvoc" ];then
    mkdir -p data/pascalvoc
    ./download.sh
fi
FLAGS_benchmark=true FLAGS_fraction_of_gpu_memory_to_use=0.0 python train.py --batch_size=64 --num_passes=1

