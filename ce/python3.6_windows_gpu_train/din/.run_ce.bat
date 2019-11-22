@echo off
set CUDA_VISIBLE_DEVICES=0
python -u train.py --config_path data/config.txt --train_dir data/paddle_train.txt --batch_size 32 --epoch_num 1 --use_cuda 1 --enable_ce --batch_num 10000 |  python _ce.py


