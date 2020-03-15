#!/bin/bash

DATA_PATH=./dataset/wmt16

train(){
    python -u main.py \
        --do_train True \
        --src_vocab_fpath $DATA_PATH/en_10000.dict \
        --trg_vocab_fpath $DATA_PATH/de_10000.dict \
        --special_token '<s>' '<e>' '<unk>' \
        --training_file $DATA_PATH/wmt16/train \
        --use_token_batch True \
        --batch_size 2048 \
        --sort_type pool \
        --pool_size 10000 \
        --print_step 1 \
        --weight_sharing False \
        --epoch 20 \
        --enable_ce True \
        --random_seed 1000 \
        --save_checkpoint "trained_ckpts" \
        --save_param "trained_params"
}

cudaid=${transformer:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid

train 1> log_1card
cat log_1card | python _ce.py

cudaid=${transformer_m:=0,1,2,3} # use 0,1,2,3 card as default
export CUDA_VISIBLE_DEVICES=$cudaid

train 1> log_4cards
cat log_4cards | python _ce.py

export CUDA_VISIBLE_DEVICES=0
python -u main.py \
        --do_predict True \
        --src_vocab_fpath $DATA_PATH/en_10000.dict \
        --trg_vocab_fpath $DATA_PATH/de_10000.dict \
        --special_token '<s>' '<e>' '<unk>' \
        --predict_file $DATA_PATH/wmt16/test \
        --batch_size 32 \
        --init_from_params  saved_models/step_final\
        --beam_size 5 \
        --max_out_len 255\
        --output_file predict.txt >infer
if [ $? -ne 0 ];then
    echo -e "transformer,infer,FAIL"
else
    echo -e "transformer,infer,SUCCESS"
fi 
