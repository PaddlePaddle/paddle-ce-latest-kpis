@echo off

set CPU_NUM=1
set NUM_THREADS=1
python train.py --train_data_path data/raw/train.txt --num_passes=1 --enable_ce | python _ce.py

set CPU_NUM=1
set NUM_THREADS=8
python train.py --train_data_path data/raw/train.txt --num_passes=1 --enable_ce | python _ce.py

set CPU_NUM=8
set NUM_THREADS=8
python train.py --train_data_path data/raw/train.txt --num_passes=1 --enable_ce| python _ce.py
