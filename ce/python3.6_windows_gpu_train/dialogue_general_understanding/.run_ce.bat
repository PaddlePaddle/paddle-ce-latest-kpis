@echo off 
set FLAGS_sync_nccl_allreduce=0
set FLAGS_eager_delete_tensor_gb=1.0

set CUDA_VISIBLE_DEVICES=0	
rem train
python -u main.py --task_name=atis_intent --use_cuda true --do_train=true  --in_tokens=true --epoch=2 --batch_size=4096  --do_lower_case=true --data_dir=./data/input/data/atis/atis_intent --bert_config_path=data/pretrain_model/uncased_L-12_H-768_A-12/bert_config.json --vocab_path="data/pretrain_model/uncased_L-12_H-768_A-12/vocab.txt" --init_from_pretrain_model=data/pretrain_model/uncased_L-12_H-768_A-12/params --save_model_path=./data/saved_models/atis_intent --save_param="params" --save_steps=100 --learning_rate=2e-5  --weight_decay=0.01 --max_seq_len=128 --print_steps=10 --use_fp16 false





