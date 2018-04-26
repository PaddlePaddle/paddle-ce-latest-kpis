#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
cudaid=${seq2seq_cudaid:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

#imdb 128
FLAGS_benchmark=true FLAGS_fraction_of_gpu_memory_to_use=0.0 python model.py --device=GPU --batch_size=128 --iterations=50 --gpu_id=$cudaid
python get_gpu_data.py --batch_size=128 --data_set=wmb
for pid in $(ps -ef | grep nvidia-smi | grep -v grep | cut -c 9-15); do
    echo $pid
    kill -9 $pid
done
