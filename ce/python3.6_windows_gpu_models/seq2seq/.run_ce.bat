@echo off
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fast_eager_deletion_mode=1
set FLAGS_fraction_of_gpu_memory_to_use=0

set CUDA_VISIBLE_DEVICES=0
rem seq2seq_base
train.py --src_lang en --tar_lang vi --attention False --num_layers 2 --hidden_size 512 --src_vocab_size 17191 --tar_vocab_size 7709 --batch_size 128 --dropout 0.2 --init_scale  0.1 --max_grad_norm 5.0 --train_data_prefix data/en-vi/train --eval_data_prefix data/en-vi/tst2012 --test_data_prefix data/en-vi/tst2013 --vocab_prefix data/en-vi/vocab --use_gpu True --model_path output --max_epoch 12 > %log_path%/seq2seq_T.log 2>&1
type %log_path%\seq2seq_T.log| grep "test ppl"|awk -F "[][]" "{print $2}"|awk "NR==1{min=$1;next}{min=min<$1?min:$1}END{print \"kpis\ttest_ppl\t\"min}"| python _ce.py
if not %errorlevel% == 0 (
        move  %log_path%\seq2seq_T.log  %log_path%\FAIL\seq2seq_T.log
        echo   seq2seq,train,FAIL  >> %log_path%\result.log
        echo   train of seq2seq failed!
) else (
        move  %log_path%\seq2seq_T.log  %log_path%\SUCCESS\seq2seq_T.log
        echo   seq2seq,train,SUCCESS  >> %log_path%\result.log
        echo   train of seq2seq successfully!
)