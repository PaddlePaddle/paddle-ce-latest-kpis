#!/bin/bash

export FLAGS_sync_nccl_allreduce=0
export FLAGS_eager_delete_tensor_gb=1.0

export CUDA_VISIBLE_DEVICES=0

pretrain_model_path="ade/data/saved_models/matching_pretrained"
if [ ! -d ${pretrain_model_path} ]
then
     mkdir ${pretrain_model_path}
fi

python -u main.py \
      --do_train=true \
      --use_cuda=true \
      --loss_type="CLS" \
      --max_seq_len=50 \
      --save_model_path="ade/data/saved_models/matching_pretrained" \
      --save_param="params" \
      --training_file="ade/data/input/data/unlabel_data/train.ids" \
      --epoch=3 \
      --print_step=1 \
      --save_step=400 \
      --batch_size=256 \
      --hidden_size=256 \
      --emb_size=256 \
      --vocab_size=484016 \
      --learning_rate=0.001 \
      --sample_pro=0.1 \
      --enable_ce="store_true" 1> log_card1
cat log_card1 | python _ce.py

#infer
python -u main.py \
      --do_predict=true \
      --use_cuda=true \
      --predict_file="ade/data/input/data/unlabel_data/test.ids" \
      --init_from_params="ade/data/saved_models/matching_pretrained/params/step_final" \
      --loss_type="CLS" \
      --output_prediction_file="ade/data/output/pretrain_matching_predict" 1>infer
if [ $? -ne 0 ];then
    echo -e "auto_dialogue_evaluation,infer,FAIL"
else
    echo -e "auto_dialogue_evaluation,infer,SUCCESS"
fi
#eval
python -u main.py \
      --do_eval=true \
      --use_cuda=true \
      --evaluation_file="ade/data/input/data/unlabel_data/test.ids" \
      --output_prediction_file="ade/data/output/pretrain_matching_predict" \
      --loss_type="CLS" 1>eval
if [ $? -ne 0 ];then
    echo -e "auto_dialogue_evaluation,eval,FAIL"
else
    echo -e "auto_dialogue_evaluation,eval,SUCCESS"
fi
