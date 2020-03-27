#!/bin/bash

# This file is only used for continuous evaluation.
# dygraph single card
export FLAGS_cudnn_deterministic=True

train () {
    python main.py \
        --model_type=bow_net \
        --do_train=True \
        --do_infer=True \
        --epoch=2 \
        --batch_size=256 \
        --ce \
        --random_seed 33
}

export CUDA_VISIBLE_DEVICES=0
train | python _ce.py
sleep 20

export CUDA_VISIBLE_DEVICES=0,1,2,3
train | python _ce.py
