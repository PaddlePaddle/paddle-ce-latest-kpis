#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
cudaid=${vgg16_cudaid:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid


#cifar10 128
FLAGS_benchmark=true FLAGS_fraction_of_gpu_memory_to_use=0.0 python model.py --device=GPU --batch_size=128 --data_set=cifar10  --iterations=300 --gpu_id=$cudaid

#flowers 32
FLAGS_benchmark=true FLAGS_fraction_of_gpu_memory_to_use=0.0 python model.py --device=GPU --batch_size=32 --data_set=flowers  --iterations=100 --gpu_id=$cudaid
