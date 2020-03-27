@echo off
set CUDA_VISIBLE_DEVICES=0
rem train
python train.py --train_dir train_data --use_cuda 0 --batch_size 50 --model_dir output --pass_num 10 > %log_path%/gru4rec_T.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\gru4rec_T.log  %log_path%\FAIL\
        echo   gru4rec,train,FAIL  >> %log_path%\result.log
        echo   training of gru4rec failed!
) else (
        move   %log_path%\gru4rec_T.log  %log_path%\SUCCESS\
        echo   gru4rec,train,SUCCESS  >> %log_path%\result.log
        echo  training of gru4rec successfully!
)
python infer.py --test_dir test_data/ --model_dir output --start_index 1 --last_index 10 --use_cuda 0 > %log_path%/gru4rec_I.log 2>&1
type gru4rec_I.log |grep model |gawk -F "[: ]" "{print $4}"|gawk "NR==1{max=$1;next}{max=max>$1?max:$1}END{print  \"kpis\ttest_recall\t\"max}"| python _ce.py
if not %errorlevel% == 0 (
        move  %log_path%/gru4rec_I.log  %log_path%\FAIL\gru4rec_I.log
        echo  gru4rec,infer,FAIL  >> %log_path%\result.log
        echo  infer of gru4rec failed!
) else (
        move  %log_path%\gru4rec_I.log  %log_path%\SUCCESS\gru4rec_I.log
        echo  gru4rec,infer,SUCCESS  >> %log_path%\result.log
        echo  infer of gru4rec successfully! 
)


