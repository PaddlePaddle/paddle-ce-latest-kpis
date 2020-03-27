#! /bin/bash
dataset=ptb
train() {
    python train.py \
        --vocab_size 10003 \
        --batch_size 32 \
        --init_scale 0.1 \
        --max_grad_norm 5.0 \
        --dataset_prefix data/${dataset}/${dataset} \
        --model_path ${dataset}_model\
        --use_gpu True \
        --max_epoch 1 \
        --enable_ce
}

cudaid=${variational_seq2seq:=0} # use 0-th card as default
export CUDA_VISIBLE_DEVICES=$cudaid
train 1> log_1card
cat log_1card | python _ce.py

python infer.py 
       --vocab_size 10003 \
        --batch_size 32 \
        --init_scale 0.1 \
        --max_grad_norm 5.0 \
        --dataset_prefix data/${dataset}/${dataset} \
        --use_gpu True \
        --reload_model ${dataset}_model/epoch_0  >infer
if [ $? -ne 0 ];then
	echo -e "variational_seq2seq,infer,FAIL"
else
	echo -e "variational_seq2seq,infer,SUCCESS"
fi

