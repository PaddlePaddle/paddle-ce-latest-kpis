@echo off
set FLAGS_cudnn_deterministic=True
set CUDA_VISIBLE_DEVICES=0
rem mnist
python train.py --epoch 1 --ce | python _ce.py







