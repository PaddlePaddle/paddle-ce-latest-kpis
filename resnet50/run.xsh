#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
cudaid=${resnet50_cudaid:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

# cifar10 128
FLAGS_benchmark=true FLAGS_fraction_of_gpu_memory_to_use=0.0 python model.py --device=GPU --batch_size=128 --data_set=cifar10 --model=resnet_cifar10 --pass_num=30 --gpu_id=$cudaid

#flowers 64
FLAGS_benchmark=true FLAGS_fraction_of_gpu_memory_to_use=0.0 python model.py --device=GPU --batch_size=64 --data_set=flowers --model=resnet_imagenet --pass_num=3 --gpu_id=$cudaid
