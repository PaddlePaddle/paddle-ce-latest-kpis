@echo off
set MKL_NUM_THREADS=1
set OMP_NUM_THREADS=1

set CPU_NUM=1
set NUM_THREADS=1
python -u train.py --use_cuda 0 --epoch_num 1 --enable_ce | python _ce.py 



