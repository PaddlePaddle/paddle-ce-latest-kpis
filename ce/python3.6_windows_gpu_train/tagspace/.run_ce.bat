@echo off
set CUDA_VISIBLE_DEVICES=0
rem train
python train.py  --use_cuda 1  --enable_ce | python _ce.py
rem infer
python infer.py --use_cuda 1 > %log_path%\tagsapce_I.log 2>&1
if %errorlevel% GTR 0 (
        move  %log_path%\tagspace_I.log  %log_path%\FAIL\tagsapce_I.log
        echo   tagspace,infer,FAIL  >> %log_path%\result.log
        echo    infer of tagspace failed!
) else (
        move  %log_path%\tagsapce_I.log  %log_path%\SUCCESS\tagsapce_I.log
        echo  tagspace,infer,SUCCESS  >> %log_path%\result.log
        echo  infer of tagspace successfully!
)
