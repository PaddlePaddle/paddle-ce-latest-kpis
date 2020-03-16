@echo off
set CUDA_VISIBLE_DEVICES=0
rem train
python -u train.py --use_cuda 1 --epoch_num 30 > %log_path%/gnn_T.log 2>&1
if not %errorlevel% == 0 (
    move %log_path%\gnn_T.log %log_pacth%\FAIL\
    echo  gnn,train,FAIL >> %log_path%\result.log
    echo  train of gnn failed!
) else (
    move %log_path%\gnn_T.log %log_path%\SUCCESS\
    echo  gnn,train,SUCCESS >> %log_path%\result.log
    echo  train of gnn successfully!
)
rem infer
python infer.py --use_cuda 1 --last_index 29 > %log_path%/gnn_I.log 2>&1
type  %log_path%/gnn_I.log |grep Recall@20 gnn_GPU_I.log |gawk -F ":" "{print $6}"|gawk "NR==1{max=$1;next}{max=max>$1?max:$1}END{print  \"kpis\ttest_recall\t\"max}"| python _ce.py 
if not %errorlevel% == 0 (
    move %log_path%\gnn_I.log %log_pacth%\FAIL\
    echo  gnn,infer,FAIL >> %log_path%\result.log
    echo  infer of gnn failed!
) else (
    move %log_path%\gnn_I.log %log_path%\SUCCESS\
    echo  gnn,infer,SUCCESS >> %log_path%\result.log
    echo  infer of gnn successfully!
)



