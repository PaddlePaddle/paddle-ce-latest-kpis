#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export FLAGS_cpu_deterministic=true
 
# CPU Test
# cifar10 16
export CPU_NUM=4
python train.py --use_gpu=false --reduce_strategy="AllReduce" --model=resnet_cifar10 --batch_size=8 --iterations=40 --pass_num=5 
python train.py --use_gpu=false --reduce_strategy="Reduce"    --model=resnet_cifar10 --batch_size=8 --iterations=40 --pass_num=5 
#flowers 16
python train.py --use_gpu=false --reduce_strategy="AllReduce" --model=resnet_imagenet --batch_size=8 --iterations=40 --pass_num=5 
python train.py --use_gpu=false --reduce_strategy="Reduce"    --model=resnet_imagenet --batch_size=8 --iterations=40 --pass_num=5 

# CPU Test
# cifar10 16
export CPU_NUM=1
python train.py --use_gpu=false --model=resnet_cifar10  --batch_size=8 --iterations=40 --pass_num=5 
python train.py --use_gpu=false --model=resnet_imagenet --batch_size=8 --iterations=40 --pass_num=5
