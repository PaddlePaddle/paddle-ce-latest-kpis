#! /bin/bash
train() {
    python train.py \
        --src_lang en --tar_lang vi \
        --attention True \
        --num_layers 2 \
        --hidden_size 512 \
        --src_vocab_size 17191 \
        --tar_vocab_size 7709 \
        --batch_size 128 \
        --dropout 0.2 \
        --init_scale  0.1 \
        --max_grad_norm 5.0 \
        --train_data_prefix data/en-vi/train \
        --eval_data_prefix data/en-vi/tst2012 \
        --test_data_prefix data/en-vi/tst2013 \
        --vocab_prefix data/en-vi/vocab \
        --use_gpu False \
        --model_path attention_models \
        --max_epoch 1 \
        --enable_ce
}

cudaid=${seq2seq_1:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid
export CPU_NUM=10
train 1> log
cat log | python _ce.py
