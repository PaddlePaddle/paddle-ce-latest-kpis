#! /bin/bash

BERT_BASE_PATH="./data/pretrained_models/uncased_L-12_H-768_A-12/"
TASK_NAME='COLA'
DATA_PATH="./data/glue_data/CoLA/"
CKPT_PATH="./data/saved_model/cola_models"

# start fine-tuning
train_single() {
  python run_classifier.py\
    --task_name ${TASK_NAME} \
    --use_cuda true \
    --do_train true \
    --do_test true \
    --batch_size 64 \
    --init_pretraining_params ${BERT_BASE_PATH}/dygraph_params/ \
    --data_dir ${DATA_PATH} \
    --vocab_path ${BERT_BASE_PATH}/vocab.txt \
    --checkpoints ${CKPT_PATH} \
    --save_steps 1000 \
    --weight_decay  0.01 \
    --warmup_proportion 0.1 \
    --validation_steps 100 \
    --epoch 3 \
    --max_seq_len 128 \
    --bert_config_path ${BERT_BASE_PATH}/bert_config.json \
    --learning_rate 5e-5 \
    --skip_steps 10 \
    --shuffle true \
    --enable_ce true
}

train_multi() {
  python -m paddle.distributed.launch --selected_gpus=$1 --log_dir ./cls_log run_classifier.py \
    --task_name ${TASK_NAME} \
    --use_cuda true \
    --use_data_parallel true \
    --do_train true \
    --do_test true \
    --batch_size 64 \
    --in_tokens false \
    --init_pretraining_params ${BERT_BASE_PATH}/dygraph_params/ \
    --data_dir ${DATA_PATH} \
    --vocab_path ${BERT_BASE_PATH}/vocab.txt \
    --checkpoints ${CKPT_PATH} \
    --save_steps 1000 \
    --weight_decay  0.01 \
    --warmup_proportion 0.1 \
    --validation_steps 100 \
    --epoch 3 \
    --max_seq_len 128 \
    --bert_config_path ${BERT_BASE_PATH}/bert_config.json \
    --learning_rate 5e-5 \
    --skip_steps 10 \
    --shuffle true \
    --enable_ce true 
}


cudaid=4 # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid
train_single 1> log_1card
cat log_1card | python _ce.py
sleep 20

cudaid=0,1,2,3 # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid
train_multi ${cudaid} 1> log_4cards
cat ./cls_log/workerlog.0 | python _ce.py
