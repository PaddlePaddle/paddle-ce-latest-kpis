@echo off
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fast_eager_deletion_mode=1
set FLAGS_fraction_of_gpu_memory_to_use=0

set CUDA_VISIBLE_DEVICES=0
python train.py  --src_lang en --tar_lang vi --attention True  --num_layers 2 --hidden_size 512 --src_vocab_size 17191 --tar_vocab_size 7709 --batch_size 128 --dropout 0.2 --init_scale  0.1 --max_grad_norm 5.0 --train_data_prefix data/en-vi/train --eval_data_prefix data/en-vi/tst2012 --test_data_prefix data/en-vi/tst2013 --vocab_prefix data/en-vi/vocab --use_gpu True --model_path ./attention_models --max_epoch 1 --enable_ce | python _ce.py

