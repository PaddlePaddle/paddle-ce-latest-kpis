#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
cudaid=${vgg16_cudaid:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

#open GC
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_memory_fraction_of_eager_deletion=0.98

#cifar10 128
#mem
FLAGS_benchmark=true FLAGS_fraction_of_gpu_memory_to_use=0.0 python model.py --device=GPU --batch_size=128 --data_set=cifar10  --iterations=300 --gpu_id=$cudaid
python get_gpu_data.py --batch_size=128 --data_set=cifar10
#speed
python model.py --device=GPU --batch_size=128 --data_set=cifar10  --iterations=300 --gpu_id=$cudaid

for pid in $(ps -ef | grep nvidia-smi | grep -v grep | cut -c 9-15); do
    echo $pid
    kill -9 $pid
done

FLOWERS_BATCH_SIZE=32
#mem
FLAGS_benchmark=true FLAGS_fraction_of_gpu_memory_to_use=0.0 python model.py --device=GPU --batch_size=${FLOWERS_BATCH_SIZE} --data_set=flowers  --iterations=100 --gpu_id=$cudaid
python get_gpu_data.py --batch_size=${FLOWERS_BATCH_SIZE} --data_set=flowers
#speed
python model.py --device=GPU --batch_size=${FLOWERS_BATCH_SIZE} --data_set=flowers  --iterations=100 --gpu_id=$cudaid

for pid in $(ps -ef | grep nvidia-smi | grep -v grep | cut -c 9-15); do
    echo $pid
    kill -9 $pid
done
