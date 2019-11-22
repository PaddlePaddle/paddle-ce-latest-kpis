@echo off
set CUDA_VISIBLE_DEVICES=0
python -u train.py --use_cuda 1 --epoch_num 1 --enable_ce | python _ce.py 



