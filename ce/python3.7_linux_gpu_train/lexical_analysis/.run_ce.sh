#!/bin/bash
export FLAGS_fraction_of_gpu_memory_to_use=0.5
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fast_eager_deletion_mode=1


train()
{
    python train.py \
        --train_data ./data/train.tsv \
        --test_data ./data/test.tsv \
        --model_save_dir ./models \
        --validation_steps 2 \
        --save_steps 10 \
        --batch_size 100 \
        --epoch 2 \
        --use_cuda true \
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
train 1> log_1card 
cat log_1card | python _ce.py
sleep 20

export CUDA_VISIBLE_DEVICES=0,1,2,3
train 1> log_4cards
cat log_4cards | python _ce.py

#eval
python eval.py \
        --batch_size 200 \
        --word_emb_dim 128 \
        --grnn_hidden_dim 128 \
        --bigru_num 2 \
        --use_cuda True \
        --init_checkpoint ./models/step_10 \
        --test_data ./data/test.tsv \
        --word_dict_path ./conf/word.dic \
        --label_dict_path ./conf/tag.dic \
        --word_rep_dict_path ./conf/q2b.dic >eval
if [ $? -ne 0 ];then
	echo -e "lac,eval,FAIL"
else
	echo -e "lac,eval,SUCCESS"
fi
#infer
python predict.py \
        --batch_size 200 \
        --word_emb_dim 128 \
        --grnn_hidden_dim 128 \
        --bigru_num 2 \
        --use_cuda True \
        --init_checkpoint ./models/step_10 \
        --infer_data ./data/infer.tsv \
        --word_dict_path ./conf/word.dic \
        --label_dict_path ./conf/tag.dic \
        --word_rep_dict_path ./conf/q2b.dic >infer
if [ $? -ne 0 ];then
        echo -e "lac,infer,FAIL"
else
        echo -e "lac,infer,SUCCESS"
fi
