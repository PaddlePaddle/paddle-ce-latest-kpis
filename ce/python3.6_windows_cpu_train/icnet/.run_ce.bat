@echo off

rem This file is only used for continuous evaluation.

set ce_mode=1
del /f /q *_factor.txt
python train.py --use_gpu=False --random_mirror=False --random_scaling=False 1> log
type log | python _ce.py
