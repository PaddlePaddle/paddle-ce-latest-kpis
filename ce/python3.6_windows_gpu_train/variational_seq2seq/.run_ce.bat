@echo off
set CUDA_VISIBLE_DEVICES=0
python train.py --vocab_size 10003 --batch_size 32 --init_scale 0.1 --max_grad_norm 5.0 --dataset_prefix data/ptb/ptb --model_path ptb_model --use_gpu True --max_epoch 1 --enable_ce | python _ce.py

