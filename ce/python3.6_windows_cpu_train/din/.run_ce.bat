@echo off

set MKL_NUM_THREADS=1
set OMP_NUM_THREADS=1

set CPU_NUM=1
set NUM_THREADS=1
rem train
python -u train.py --config_path data/config.txt --train_dir data/paddle_train.txt --batch_size 32 --epoch_num 1 --use_cuda 0 --enable_ce --batch_num 50000 |  python _ce.py
rem infer
python infer.py --model_path din_amazon/global_step_50000 --test_path data/paddle_test.txt --use_cuda 0 >%log_path%/din_I.log 2>&1
if not %errorlevel% == 0 (
     move %log_path%\din_I.log %log_path%\FAIL\
     echo  din,infer,FAIL >> %log_path%\result.log
     echo  infer of din failed!
) else (
        move %log_path%\din_I.log %log_path%\SUCCESS\
        echo  din,infer,SUCCESS >> %log_path%\result.log
        echo  infer of din successfully!
)

