@echo off
set CUDA_VISIBLE_DEVICES=0
rem train
python -u train.py --use_cuda 1 --epoch_num 1 --enable_ce | python _ce.py 
rem infer
python infer.py --last_index 1 --use_cuda 1 >%log_path%/gnn_I.log 2>&1
if not %errorlevel% == 0 (
    move %log_path%\gnn_I.log %log_pacth%\FAIL\gnn_I.log
    echo  gnn,infer,FAIL >> %log_path%\result.log
    echo  infer of gnn failed!
) else (
    move %log_path%\gnn_I.log %log_path%\SUCCESS\gnn_I.log
    echo  gnn,infer,SUCCESS >> %log_path%\result.log
    echo  infer of gnn successfully!
)



