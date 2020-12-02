#! /bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7


# start fine-tuning
train_single() {
 python -u train.py \
  --epoch 2 \
  --src_vocab_fpath wmt16_ende_data_bpe/vocab_all.bpe.32000 \
  --trg_vocab_fpath wmt16_ende_data_bpe/vocab_all.bpe.32000 \
  --special_token '<s>' '<e>' '<unk>' \
  --training_file wmt16_ende_data_bpe/train_file \
  --validation_file wmt16_ende_data_bpe/newstest2014.tok.bpe.32000.en-de \
  --batch_size 4096 \
  --print_step 10 \
  --use_cuda True \
  --random_seed 102 \
  --enable_ce true
}

train_multi() {
 python -m paddle.distributed.launch --started_port 8999 --gpus=$1 --log_dir ./mylog train.py \
  --epoch 2 \
  --src_vocab_fpath wmt16_ende_data_bpe/vocab_all.bpe.32000 \
  --trg_vocab_fpath wmt16_ende_data_bpe/vocab_all.bpe.32000 \
  --special_token '<s>' '<e>' '<unk>' \
  --training_file wmt16_ende_data_bpe/train_file \
  --validation_file wmt16_ende_data_bpe/newstest2014.tok.bpe.32000.en-de \
  --batch_size 4096 \
  --print_step 10 \
  --use_cuda True \
  --save_step 10000 \
  --random_seed 102 \
  --enable_ce true
}


cudaid=0 # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid
train_single 1> log_1card
cat log_1card | python _ce.py
sleep 20

cudaid=0,1,2,3 # use 0-th card as default
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
train_multi ${cudaid} 1> log_4cards
cat ./mylog/workerlog.0 | python _ce.py
