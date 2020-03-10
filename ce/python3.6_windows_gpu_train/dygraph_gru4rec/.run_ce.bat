@echo off

set CUDA_VISIBLE_DEVICES=0
set ce_mode=1
rem train
python -u gru4rec_dy.py  --data_path data --model_type gru4rec --use_gpu true  --ce | python _ce.py

