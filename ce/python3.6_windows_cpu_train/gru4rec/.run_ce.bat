@echo off

set MKL_NUM_THREADS=1
set OMP_NUM_THREADS=1
set CPU_NUM=1
set NUM_THREADS=1
set ce_mode=1
python train.py --train_dir train_data --use_cuda 0 --pass_num 1 --model_dir output --pass_num 2 --enable_ce --step_num 10 | python _ce.py

