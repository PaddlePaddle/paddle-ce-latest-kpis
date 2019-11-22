@echo off

set MKL_NUM_THREADS=1
set OMP_NUM_THREADS=1

set CPU_NUM=1
set NUM_THREADS=1
python -u train.py --config_path data/config.txt --train_dir data/paddle_train.txt --batch_size 32 --epoch_num 1 --use_cuda 0 --enable_ce --batch_num 10000 |  python _ce.py


