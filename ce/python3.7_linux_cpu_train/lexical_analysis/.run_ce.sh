#!/bin/bash

train()
{
    python train.py \
        --train_data ./data/train.tsv \
        --test_data ./data/test.tsv \
        --model_save_dir ./models \
        --validation_steps 2 \
        --save_steps 5000 \
        --batch_size 100 \
        --epoch 1 \
        --use_cuda false \
        --traindata_shuffle_buffer 200000 \
        --word_emb_dim 768 \
        --grnn_hidden_dim 768 \
        --bigru_num 2 \
        --base_learning_rate 1e-3 \
        --emb_learning_rate 5 \
        --crf_learning_rate 0.2 \
        --word_dict_path ./conf/word.dic \
        --label_dict_path ./conf/tag.dic \
        --word_rep_dict_path ./conf/q2b.dic \
        --enable_ce true
}

export CUDA_VISIBLE_DEVICES=0
export CPU_NUM=10
train 1> log 
cat log | python _ce.py
