@echo off

set MKL_NUM_THREADS=1
set OMP_NUM_THREADS=1

set CPU_NUM=1
set NUM_THREADS=1
rem train
python train.py --enable_ce --epochs=1 | python _ce.py
rem infer      
python infer.py > %log_path%/multiview_I.log 2>&1
if not %errorlevel% == 0 (
   move  %log_path%\multiview_I.log  %log_path%\FAIL\multiview_I.log
   echo   multiview,infer,FAIL  >> %log_path%\result.log
   echo   infer of multiview failed!
 ) else (
     move  %log_path%\multiview_I.log  %log_path%\SUCCESS\multiview_I.log
    echo   multiview,infer,SUCCESS  >> %log_path%\result.log
    echo   infer of multiview successful
)

