@echo off
set FLAGS_cudnn_deterministic=True
set CUDA_VISIBLE_DEVICES=0
rem ptb_lm
python ptb_dy.py --data_path data/simple-examples/data --ce --model_type small --use_gpu True | python _ce.py
              







