@echo off
set CUDA_VISIBLE_DEVICES=0
set ce_mode=1
python train.py --batch_size 8 --dataset coco --num_epochs=1  --enable_ce true --use_gpu false | python _ce.py