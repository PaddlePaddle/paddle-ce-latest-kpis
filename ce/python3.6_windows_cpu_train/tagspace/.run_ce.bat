@echo off

set MKL_NUM_THREADS=1
set OMP_NUM_THREADS=1


set CPU_NUM=1
set NUM_THREADS=1
rem train
python train.py  --use_cuda 0 --enable_ce | python _ce.py
rem infer
python infer.py --use_cuda 0 > %log_path%\tagsapce_I.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\tagspace_I.log  %log_path%\FAIL\tagsapce_I.log
        echo   tagspace,infer,FAIL  >> %log_path%\result.log
        echo    infer of tagspace failed!
) else (
        move  %log_path%\tagsapce_I.log  %log_path%\SUCCESS\tagsapce_I.log
        echo  tagspace,infer,SUCCESS  >> %log_path%\result.log
        echo  infer of tagspace successfully!
)

