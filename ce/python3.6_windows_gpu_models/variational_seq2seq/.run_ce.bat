@echo off
set CUDA_VISIBLE_DEVICES=0
rem train

python train.py --vocab_size 10003 --batch_size 32 --init_scale 0.1 --max_grad_norm 5.0 --dataset_prefix data/ptb/ptb --model_path ptb_model --use_gpu True --max_epoch 50 > %log_path%/variational_seq2seq_ptb_T.log 2>&1

