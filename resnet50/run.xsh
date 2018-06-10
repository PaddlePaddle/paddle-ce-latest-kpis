#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

arr[0]='0'
arr[1]='0,1,2,3'
if [[ `nvidia-smi  --list-gpus| wc -l` = 8 ]]; then
    #agent has 4 card no need to run 8 card scene
    arr[2]='0,1,2,3,4,5,6,7'
else 
    cp -rf latest_kpis/*card8* ../ 
fi

for var in ${arr[@]};
do
    echo $var
    cudaid=$var
    export CUDA_VISIBLE_DEVICES=$cudaid
    echo $CUDA_VISIBLE_DEVICES

    #FLAGS_benchmark=true FLAGS_fraction_of_gpu_memory_to_use=0.0 
    python model.py --device=GPU --batch_size=128 --data_set=cifar10 --model=resnet_cifar10 --pass_num=3  --gpu_id $cudaid

    #FLAGS_benchmark=true FLAGS_fraction_of_gpu_memory_to_use=0.0 
    python model.py --device=GPU --batch_size=32 --data_set=flowers --model=resnet_imagenet --pass_num=3  --use_fake_data --gpu_id $cudaid --iterations 20

    for pid in $(ps -ef | grep nvidia-smi | grep -v grep | cut -c 9-15); do
        echo $pid
        kill -9 $pid
    done
done
