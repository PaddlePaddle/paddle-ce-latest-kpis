@echo off
set CUDA_VISIBLE_DEVICES=0
python train.py --train_dir train_data --use_cuda 1 --batch_size 50 --model_dir model_output --epochs 2 --enable_ce --step_num 1000 | python _ce.py

