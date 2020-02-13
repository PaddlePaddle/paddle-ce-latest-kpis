@echo off
rem This file is only used for continuous evaluation.
set CUDA_VISIBLE_DEVICES=0
python train.py --model_save_dir=output/ --pretrain=weights/darknet53 --data_dir=dataset/coco --max_iter=200 --snapshot_iter 100 --use_multiprocess_reader false --batch_size 4 --use_gpu true --enable_ce True | python _ce.py

