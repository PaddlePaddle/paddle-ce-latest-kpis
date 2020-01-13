#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
python train_elem.py --model=ResNet50 --test_iter_step=10 --loss_name=arcmargin --arc_scale=80.0 --arc_margin=0.15 --arc_easy_margin=False --total_iter_num=10 --model_save_dir=output --save_iter_step=1 --enable_ce=True 1>log
cat log | python _ce.py
