@echo off

set MKL_NUM_THREADS=1
set OMP_NUM_THREADS=1

set CPU_NUM=1
set NUM_THREADS=1
python train.py --enable_ce --epochs=1 | python _ce.py


