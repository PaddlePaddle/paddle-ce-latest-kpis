@echo off
set CUDA_VISIBLE_DEVICES=0
python train.py  --use_cuda 1  --enable_ce | python _ce.py

