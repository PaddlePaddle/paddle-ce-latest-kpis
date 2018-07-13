#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
# GPU Test
cudaid=${resnet50_cudaid:=0,1,2,3} # use 0-th card as default
# export CUDA_VISIBLE_DEVICES=$cudaid

# cifar10 128
env CUDA_VISIBLE_DEVICES=$cudaid FLAGS_benchmark=true FLAGS_fraction_of_gpu_memory_to_use=0.0 \
    python train.py --use_gpu=true --reduce_strategy="AllReduce" --batch_size=128 --data_set=cifar10 --model=resnet_cifar10 --pass_num=30 --gpu_id=$cudaid
python get_gpu_data.py --batch_size=128 --data_set=cifar10
env CUDA_VISIBLE_DEVICES=$cudaid FLAGS_benchmark=true FLAGS_fraction_of_gpu_memory_to_use=0.0 \
    python train.py --use_gpu=true --reduce_strategy="Reduce"  --batch_size=128 --data_set=cifar10 --model=resnet_cifar10 --pass_num=30 --gpu_id=$cudaid
python get_gpu_data.py --batch_size=128 --data_set=cifar10

#flowers 64
env CUDA_VISIBLE_DEVICES=$cudaid FLAGS_benchmark=true FLAGS_fraction_of_gpu_memory_to_use=0.0 \
    python train.py --use_gpu=true --reduce_strategy="AllReduce" --batch_size=64 --data_set=flowers --model=resnet_imagenet --pass_num=30 --gpu_id=$cudaid
python get_gpu_data.py --batch_size=64 --data_set=flowers
env CUDA_VISIBLE_DEVICES=$cudaid FLAGS_benchmark=true FLAGS_fraction_of_gpu_memory_to_use=0.0 \
    python train.py --use_gpu=true --reduce_strategy="Reduce" --batch_size=64 --data_set=flowers --model=resnet_imagenet --pass_num=30 --gpu_id=$cudaid
python get_gpu_data.py --batch_size=64 --data_set=flowers

for pid in $(ps -ef | grep nvidia-smi | grep -v grep | cut -c 9-15); do
    echo $pid
    kill -9 $pid
done

# CPU Test

# cifar10 128
env CPU_NUM=4 FLAGS_benchmark=true FLAGS_fraction_of_gpu_memory_to_use=0.0 \
    python train.py --use_gpu=false --reduce_strategy="AllReduce" --batch_size=128 --data_set=cifar10 --model=resnet_cifar10  --pass_num=30  --gpu_id=$cudaid
python get_gpu_data.py --batch_size=128 --data_set=cifar10
env CPU_NUM=4 FLAGS_benchmark=true FLAGS_fraction_of_gpu_memory_to_use=0.0 \
    python train.py --use_gpu=false --reduce_strategy="Reduce" --batch_size=128 --data_set=cifar10 --model=resnet_cifar10  --pass_num=30  --gpu_id=$cudaid
python get_gpu_data.py --batch_size=128 --data_set=cifar10

#flowers 64
env CPU_NUM=4 FLAGS_benchmark=true FLAGS_fraction_of_gpu_memory_to_use=0.0 \
    python train.py --use_gpu=false --reduce_strategy="AllReduce" --batch_size=64 --data_set=flowers --model=resnet_imagenet --pass_num=30 --gpu_id=$cudaid
python get_gpu_data.py --batch_size=64 --data_set=flowers
env CPU_NUM=4 FLAGS_benchmark=true FLAGS_fraction_of_gpu_memory_to_use=0.0 \
    python train.py --use_gpu=false --reduce_strategy="Reduce" --batch_size=64 --data_set=flowers --model=resnet_imagenet --pass_num=30 --gpu_id=$cudaid
python get_gpu_data.py --batch_size=64 --data_set=flowers