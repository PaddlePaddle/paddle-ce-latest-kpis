#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1


cudaid=${gnn:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python -u train.py --use_cuda 1 --epoch_num 5 --enable_ce 1> log
cat log | python _ce.py 

#infer
python infer.py --last_index 0 1>infer
if [ $? -ne 0 ];then
    echo -e "gnn,infer,FAIL"
else
    echo -e "gnn,infer,SUCCESS"
fi

