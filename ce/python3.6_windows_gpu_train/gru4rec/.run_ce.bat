@echo off

set CUDA_VISIBLE_DEVICES=0
set ce_mode=1
python train.py --train_dir train_data --use_cuda 1 --pass_num 1 --model_dir output --pass_num 2 --enable_ce --step_num 1000 | python _ce.py

