@echo off
set CUDA_VISIBLE_DEVICES=0
rem train
python train.py --vocab_size 10003 --batch_size 32 --init_scale 0.1 --max_grad_norm 5.0 --dataset_prefix data/ptb/ptb --model_path ptb_model --use_gpu True --max_epoch 1 --enable_ce | python _ce.py
rem infer
python infer.py --vocab_size 10003 --batch_size 32  --init_scale 0.1  --max_grad_norm 5.0 --dataset_prefix data/ptb/ptb --use_gpu True  --reload_model ptb_model/epoch_0 > %log_path%/variational_seq2seq_ptb_I.log 2>&1
if %errorlevel% GTR 0 (
        move  %log_path%\variational_seq2seq_ptb_I.log  %log_path%\FAIL\variational_seq2seq_ptb_I.log
        echo   variational_seq2seq_ptb,infer,FAIL  >> %log_path%\result.log
        echo   infer of variational_seq2seq_ptb failed!
) else (
        move  %log_path%\variational_seq2seq_ptb_I.log  %log_path%\SUCCESS\variational_seq2seq_ptb_I.log
        echo   variational_seq2seq_ptb,infer,SUCCESS  >> %log_path%\result.log
        echo   infer of variational_seq2seq_ptb successfully!
)
