@echo off
set CUDA_VISIBLE_DEVICES=0

python train.py --data_path data/simple-examples/data/  --model_type small --rnn_model static  --use_gpu True --max_epoch 10 > %log_path%/language_model_T.log 2>&1
type %log_path%\language_model_T.log| grep "Test ppl" |gawk "{print \"kpis\ttest_ppl\t\"$NF}"|python _ce.py
if not %errorlevel% == 0 (
        move  %log_path%\language_model_T.log  %log_path%\FAIL\language_model_T.log
        echo   language_model,train,FAIL  >> %log_path%\result.log
        echo   train of language_model failed!
) else (
        move  %log_path%\language_model_T.log  %log_path%\SUCCESS\language_model_T.log
        echo   language_model,train,SUCCESS  >> %log_path%\result.log
        echo   train of language_model successfully!
)