@echo off

set CPU_NUM=1
set NUM_THREADS=1
rem train
python train.py --train_data_path data/raw/train.txt --num_passes=1 --enable_ce | python _ce.py 
rem infer
python infer.py --model_path models/pass-0 --data_path data/raw/train.txt > %log_path%/dnn_I.log 2>&1
if not %errorlevel% == 0 (
        move %log_path%\dnn_I.log %log_path%\FAIL\
        echo dnn,infer,FAIL >> %log_path%\result.log
        echo infering of dnn failed!
) else (
        move  %log_path%\dnn_I.log  %log_path%\SUCCESS\
        echo  dnn,infer,SUCCESS >>%log_path%\result.log
        echo  infering of dnn successfully!
 )
