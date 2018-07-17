#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
cudaid=${resnet50_cudaid:=0,1,2,3} 
export CUDA_VISIBLE_DEVICES=$cudaid 
# export FLAGS_benchmark=true 
# export FLAGS_fraction_of_gpu_memory_to_use=0.0
export CPU_NUM=4
# GPU Test
# cifar10 128
python train.py --use_gpu=true --reduce_strategy="AllReduce" --batch_size=128 --model=resnet_cifar10 --pass_num=5 --gpu_id=$cudaid
python get_gpu_data.py --batch_size=128 --data_set=cifar10 --reduce_strategy="AllReduce"

python train.py --use_gpu=true --reduce_strategy="Reduce"  --batch_size=128 --model=resnet_cifar10 --pass_num=5 --gpu_id=$cudaid
python get_gpu_data.py --batch_size=128 --data_set=cifar10 --reduce_strategy="Reduce"
for pid in $(ps -ef | grep nvidia-smi | grep -v grep | cut -c 9-15); do
    echo $pid
    kill -9 $pid
done

#flowers 64
python train.py --use_gpu=true --reduce_strategy="AllReduce" --batch_size=64 --model=resnet_imagenet --pass_num=5 --gpu_id=$cudaid
python get_gpu_data.py --batch_size=64 --data_set=flowers --reduce_strategy="AllReduce"
python train.py --use_gpu=true --reduce_strategy="Reduce" --batch_size=64 --model=resnet_imagenet --pass_num=5 --gpu_id=$cudaid
python get_gpu_data.py --batch_size=64 --data_set=flowers  --reduce_strategy="Reduce"
for pid in $(ps -ef | grep nvidia-smi | grep -v grep | cut -c 9-15); do
    echo $pid
    kill -9 $pid
done

# CPU Test
# cifar10 16
python train.py --use_gpu=false --reduce_strategy="AllReduce" --batch_size=8 --model=resnet_cifar10  --pass_num=5 
python train.py --use_gpu=false --reduce_strategy="Reduce" --batch_size=8 --model=resnet_cifar10  --pass_num=5 

#flowers 16
python train.py --use_gpu=false --reduce_strategy="AllReduce" --batch_size=8 --model=resnet_imagenet --pass_num=5 
python train.py --use_gpu=false --reduce_strategy="Reduce" --batch_size=8 --model=resnet_imagenet --pass_num=5 


# Single card
cudaid=${resnet50_cudaid:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid 
export CPU_NUM=1
# GPU Test
# cifar10 128
python train.py --use_gpu=true --batch_size=128 --model=resnet_cifar10 --pass_num=5 --gpu_id=$cudaid
python get_gpu_data.py --batch_size=128 --data_set=cifar10
for pid in $(ps -ef | grep nvidia-smi | grep -v grep | cut -c 9-15); do
    echo $pid
    kill -9 $pid
done

#flowers 64
python train.py --use_gpu=true --batch_size=64 --model=resnet_imagenet --pass_num=5 --gpu_id=$cudaid
python get_gpu_data.py --batch_size=64 --data_set=flowers
for pid in $(ps -ef | grep nvidia-smi | grep -v grep | cut -c 9-15); do
    echo $pid
    kill -9 $pid
done

# CPU Test
# cifar10 16
python train.py --use_gpu=false --batch_size=8 --model=resnet_cifar10  --pass_num=5 

#flowers 16
python train.py --use_gpu=false --batch_size=8 --model=resnet_imagenet --pass_num=5