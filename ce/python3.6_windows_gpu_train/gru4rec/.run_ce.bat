@echo off

set CUDA_VISIBLE_DEVICES=0
set ce_mode=1
rem train
python train.py --train_dir train_data --use_cuda 1 --pass_num 1 --model_dir output --pass_num 2 --enable_ce --step_num 1000 | python _ce.py

python train_sample_neg.py --loss bpr --use_cuda 1 --pass_num 1 > %log_path%/gru4rec_neg_bpr_T.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\gru4rec_neg_bpr_T.log  %log_path%\FAIL\
        echo   gru4rec_neg_bpr,train,FAIL  >> %log_path%\result.log
        echo   training of gru4rec_neg_bpr failed!
) else (
        move   %log_path%\gru4rec_neg_bpr_T.log  %log_path%\SUCCESS\
        echo   gru4rec_neg_bpr,train,SUCCESS  >> %log_path%\result.log
        echo  training of gru4rec_neg_bpr successfully!
)
python train_sample_neg.py --loss ce --use_cuda 1 --pass_num 1 > %log_path%/gru4rec_neg_ce_T.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\gru4rec_neg_ce_T.log  %log_path%\FAIL\
        echo   gru4rec_neg_ce,train,FAIL  >> %log_path%\result.log
        echo   training of gru4rec_neg_ce failed!
) else (
        move   %log_path%\gru4rec_neg_ce_T.log  %log_path%\SUCCESS\
        echo   gru4rec_neg_ce,train,SUCCESS  >> %log_path%\result.log
        echo  training of gru4rec_neg_ce successfully!
)
python infer.py --test_dir test_data/ --model_dir output/ --start_index 1 --last_index 1 --use_cuda 1 > %log_path%/gru4rec_I.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\gru4rec_I.log  %log_path%\FAIL\gru4rec_I.log
        echo  gru4rec,infer,FAIL  >> %log_path%\result.log
        echo  infer of gru4rec failed!
) else (
        move  %log_path%\gru4rec_I.log  %log_path%\SUCCESS\gru4rec_I.log
        echo  gru4rec,infer,SUCCESS  >> %log_path%\result.log
        echo  infer of gru4rec successfully! 
)

