#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1


cudaid=${face_detection:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python -u train.py --config_path 'data/config.txt' --train_dir 'data/paddle_train.txt' --batch_size 32 --epoch_num 1 --use_cuda 1 --enable_ce --batch_num 10000 1> log_1card
cat log_1card |  python _ce.py


cudaid=${face_detection_4:=0,1,2,3} # use 0,1,2,3 card as default
export CUDA_VISIBLE_DEVICES=$cudaid

FLAGS_benchmark=true  python -u train.py --config_path 'data/config.txt' --train_dir 'data/paddle_train.txt' --batch_size 32 --epoch_num 1 --use_cuda 1 --enable_ce --batch_num 10000 1> log_4cards
cat log_4cards |  python _ce.py

#infer
python infer.py --model_path 'din_amazon/global_step_50000' --test_path 'data/paddle_test.txt' --use_cuda 1 >infer
if [ $? -ne 0 ];then
    echo -e "din,infer,FAIL"
else
    echo -e "din,infer,SUCCESS"
fi
