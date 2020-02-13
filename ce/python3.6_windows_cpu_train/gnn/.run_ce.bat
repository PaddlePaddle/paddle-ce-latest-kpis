@echo off
set MKL_NUM_THREADS=1
set OMP_NUM_THREADS=1

set CPU_NUM=1
set NUM_THREADS=1
rem train
python -u train.py --use_cuda 0 --epoch_num 1 --enable_ce | python _ce.py 
rem infer
python infer.py --last_index 1 --use_cuda 0 >%log_path%/gnn_I.log 2>&1
if not %errorlevel% == 0 (
    move %log_path%\gnn_I.log %log_pacth%\FAIL\
    echo  gnn,infer,FAIL >> %log_path%\result.log
    echo  infer of gnn failed!
) else (
    move %log_path%\gnn_I.log %log_path%\SUCCESS\
    echo  gnn,infer,SUCCESS >> %log_path%\result.log
    echo  infer of gnn successfully!
)



