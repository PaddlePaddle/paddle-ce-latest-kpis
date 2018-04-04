#!/usr/bin/env xonsh
import os
import sys

os.environ["MKL_NUM_THREADS"]=1
os.environ["OMP_NUM_THREADS"]=1
os.environ["CUDA_VISIBLE_DEVICES"]=3

model_file = 'model.py'

# cifar10 128
FLAGS_benchmark=true FLAGS_fraction_of_gpu_memory_to_use=0.0 python @(model_file) --device=GPU --batch_size=128 --data_set=cifar10 --model=resnet_cifar10 --pass_num=30
python get_gpu_data.py --batch_size=128 --data_set=cifar10

#flowers 64
FLAGS_benchmark=true FLAGS_fraction_of_gpu_memory_to_use=0.0 python @(model_file) --device=GPU --batch_size=64 --data_set=flowers --model=resnet_imagenet --pass_num=3
python get_gpu_data.py --batch_size=64 --data_set=flowers
