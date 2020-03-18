#!/bin/bash
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

export CPU_NUM=1
export NUM_THREADS=1
python train.py --enable_ce --epochs=1 | python _ce.py


