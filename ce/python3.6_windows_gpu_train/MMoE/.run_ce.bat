@echo off

set CUDA_VISIBLE_DEVICES=0
rem train
python mmoe_train.py > MMoE.log 2>&1

set sed="C:\Program Files\Git\usr\bin\sed.exe"
%sed% -n '101p' mmoe.log |gawk  "{print \"kpis\ttrain_loss\t\"$2}" |gawk -F "[][]" "{print \"kpis\ttrain_loss\t\"$2}" | python _ce.py

