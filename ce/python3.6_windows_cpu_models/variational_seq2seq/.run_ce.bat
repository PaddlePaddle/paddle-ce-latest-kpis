@echo off
set CUDA_VISIBLE_DEVICES=0
rem train

python train.py --vocab_size 10003 --batch_size 32 --init_scale 0.1 --max_grad_norm 5.0 --dataset_prefix data/ptb/ptb --model_path ptb_model --use_gpu false --max_epoch 50 > %log_path%/variational_seq2seq_ptb_T.log 2>&1
type %log_path%\variational_seq2seq_ptb_T.log|grep "best testing nll"|awk -F "[:, ]" "{print \"kpis\ttest_nll\t\"$5}" | python _ce.py
type %log_path%\variational_seq2seq_ptb_T.log|grep "best testing nll"|awk -F "[:, ]" "{print \"kpis\ttest_ppl\t\"$NF}" | python _ce.py
if not %errorlevel% == 0 (
        move  %log_path%\variational_seq2seq_ptb_T.log  %log_path%\FAIL\variational_seq2seq_ptb_T.log
        echo   variational_seq2seq_ptb,train,FAIL  >> %log_path%\result.log
        echo   train of variational_seq2seq_ptb failed!
) else (
        move  %log_path%\variational_seq2seq_ptb_T.log  %log_path%\SUCCESS\variational_seq2seq_ptb_T.log
        echo   variational_seq2seq_ptb,train,SUCCESS  >> %log_path%\result.log
        echo   train of variational_seq2seq_ptb successfully!
)