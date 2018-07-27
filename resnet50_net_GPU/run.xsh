#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
cudaid=${resnet50_cudaid:=0,1,2,3} 
export CUDA_VISIBLE_DEVICES=$cudaid 
# GPU Test
# cifar10 128
python train.py --use_gpu=true --reduce_strategy="AllReduce" --model=resnet_cifar10 --batch_size=128 --pass_num=5 --gpu_id=$cudaid
python get_gpu_data.py --batch_size=128 --data_set=cifar10 --reduce_strategy="AllReduce"
for pid in $(ps -ef | grep nvidia-smi | grep -v grep | cut -c 9-15); do
    echo $pid
    kill -9 $pid
done

python train.py --use_gpu=true --reduce_strategy="Reduce"    --model=resnet_cifar10 --batch_size=128 --pass_num=5 --gpu_id=$cudaid
python get_gpu_data.py --batch_size=128 --data_set=cifar10 --reduce_strategy="Reduce"
for pid in $(ps -ef | grep nvidia-smi | grep -v grep | cut -c 9-15); do
    echo $pid
    kill -9 $pid
done

# GPU Test
#flowers 64
python train.py --use_gpu=true --reduce_strategy="AllReduce" --model=resnet_imagenet --batch_size=64 --pass_num=5 --gpu_id=$cudaid
python get_gpu_data.py --batch_size=64 --data_set=flowers --reduce_strategy="AllReduce"
for pid in $(ps -ef | grep nvidia-smi | grep -v grep | cut -c 9-15); do
    echo $pid
    kill -9 $pid
done
python train.py --use_gpu=true --reduce_strategy="Reduce"  --model=resnet_imagenet --batch_size=64 --pass_num=5 --gpu_id=$cudaid
python get_gpu_data.py --batch_size=64 --data_set=flowers  --reduce_strategy="Reduce"
for pid in $(ps -ef | grep nvidia-smi | grep -v grep | cut -c 9-15); do
    echo $pid
    kill -9 $pid
done

# Single card
single_cudaid=${resnet50_single_cudaid:=0}
export CUDA_VISIBLE_DEVICES=$single_cudaid 

# GPU Test
# cifar10 128
python train.py --use_gpu=true --model=resnet_cifar10 --batch_size=128 --pass_num=5 --gpu_id=$single_cudaid
python get_gpu_data.py --batch_size=128 --data_set=cifar10
for pid in $(ps -ef | grep nvidia-smi | grep -v grep | cut -c 9-15); do
    echo $pid
    kill -9 $pid
done

#flowers 64
python train.py --use_gpu=true --model=resnet_imagenet --batch_size=64 --pass_num=5 --gpu_id=$single_cudaid
python get_gpu_data.py --batch_size=64 --data_set=flowers
for pid in $(ps -ef | grep nvidia-smi | grep -v grep | cut -c 9-15); do
    echo $pid
    kill -9 $pid
done