@echo off
set CUDA_VISIBLE_DEVICES=0
rem train
python train.py --train_dir train_data --use_cuda 1 --batch_size 50 --model_dir model_output --epochs 2 --enable_ce --step_num 1000 | python _ce.py
rem infer
python infer.py --test_dir test_data --use_cuda 1 --batch_size 50 --model_dir model_output --start_index 1 --last_index 1 > %log_path%/ssr_I.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\ssr_I.log  %log_path%\FAIL\ssr_I.log
        echo   ssr,infer,FAIL  >> %log_path%\result.log
        echo   infer of ssr failed!
) else (
        move  %log_path%\ssr_I.log  %log_path%\SUCCESS\ssr_I.log
        echo   ssr,infer,SUCCESS  >> %log_path%\result.log
        echo   infer of ssr successfully!
 )

