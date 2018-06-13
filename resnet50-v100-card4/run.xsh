#!/bin/bash


cudaid='0,1,2,3'
export CUDA_VISIBLE_DEVICES=$cudaid
echo $CUDA_VISIBLE_DEVICES
#FLAGS_benchmark=true FLAGS_fraction_of_gpu_memory_to_use=0.0 
python ../resnet50/model.py --device=GPU --batch_size=128 --data_set=cifar10 --model=resnet_cifar10 --pass_num=3  --gpu_id $cudaid
for pid in $(ps -ef | grep nvidia-smi | grep -v grep | cut -c 9-15); do
    echo $pid
    kill -9 $pid
done

#FLAGS_benchmark=true FLAGS_fraction_of_gpu_memory_to_use=0.0 
python ../resnet50/model.py --device=GPU --batch_size=32 --data_set=flowers --model=resnet_imagenet --pass_num=3  --use_fake_data --gpu_id $cudaid --iterations 20
for pid in $(ps -ef | grep nvidia-smi | grep -v grep | cut -c 9-15); do
    echo $pid
    kill -9 $pid
done
