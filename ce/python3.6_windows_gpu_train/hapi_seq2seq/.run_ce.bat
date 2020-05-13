@echo off
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fast_eager_deletion_mode=1
set FLAGS_fraction_of_gpu_memory_to_use=0

set CUDA_VISIBLE_DEVICES=0
rem attention
python train.py  --src_lang en --tar_lang vi --attention True  --num_layers 2 --hidden_size 512 --src_vocab_size 17191 --tar_vocab_size 7709 --batch_size 128 --dropout 0.2 --init_scale  0.1 --max_grad_norm 5.0 --train_data_prefix data/en-vi/train --eval_data_prefix data/en-vi/tst2012 --test_data_prefix data/en-vi/tst2013 --vocab_prefix data/en-vi/vocab --use_gpu True --model_path ./attention_models --max_epoch 1 --enable_ce >hapi_seq2seq.log 
type  hapi_seq2seq.log |grep "step 1042/1042"|gawk -F "[:-] " "END{print \"kpis\ttrain_ppl\t\"$3}"| python _ce.py
type  hapi_seq2seq.log |grep "step 13/13"|gawk -F "[:-] " "END{print \"kpis\teval_ppl\t\"$3}" | python _ce.py
rem predict
python predict.py  --attention True --src_lang en --tar_lang vi --num_layers 2 --hidden_size 512 --src_vocab_size 17191 --tar_vocab_size 7709 --batch_size 128 --dropout 0.2 --init_scale  0.1 --max_grad_norm 5.0 --vocab_prefix data/en-vi/vocab --infer_file data/en-vi/tst2013.en --reload_model attention_models/0 --infer_output_file predict_output.txt --beam_size 10 --use_gpu True > %log_path%/seq2seq_attention_I.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\seq2seq_attention_I.log  %log_path%\FAIL\seq2seq_attention_I.log
        echo   seq2seq_attention,predict,FAIL  >> %log_path%\result.log
        echo   predict of seq2seq_attention failed!
) else (
        move  %log_path%\seq2seq_attention_I.log  %log_path%\SUCCESS\seq2seq_attention_I.log
        echo   seq2seq_attention,predict,SUCCESS  >> %log_path%\result.log
        echo   predict of seq2seq_attention successfully!
)

rem seq2seq_base
python train.py  --src_lang en --tar_lang vi --attention False  --num_layers 2 --hidden_size 512 --src_vocab_size 17191 --tar_vocab_size 7709 --batch_size 128 --dropout 0.2 --init_scale  0.1 --max_grad_norm 5.0 --train_data_prefix data/en-vi/train --eval_data_prefix data/en-vi/tst2012 --test_data_prefix data/en-vi/tst2013 --vocab_prefix data/en-vi/vocab --use_gpu True --model_path ./base_models --max_epoch 1 > %log_path%/seq2seq_base_T.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\seq2seq_base_T.log  %log_path%\FAIL\seq2seq_base_T.log
        echo   seq2seq_base,train,FAIL  >> %log_path%\result.log
        echo  training of seq2seq_base failed!
) else (
        move  %log_path%\seq2seq_base_T.log  %log_path%\SUCCESS\seq2seq_base_T.log
        echo   seq2seq_base,train,SUCCESS  >> %log_path%\result.log
        echo   training of seq2seq_base successfully!
 )
rem predict
python predict.py  --attention False --src_lang en --tar_lang vi --num_layers 2 --hidden_size 512 --src_vocab_size 17191 --tar_vocab_size 7709 --batch_size 128 --dropout 0.2 --init_scale  0.1 --max_grad_norm 5.0 --vocab_prefix data/en-vi/vocab --infer_file data/en-vi/tst2013.en --reload_model base_models/0 --infer_output_file predict_output.txt --beam_size 10 --use_gpu True > %log_path%/seq2seq_base_I.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\seq2seq_base_I.log  %log_path%\FAIL\seq2seq_base_I.log
        echo   seq2seq_base,predict,FAIL  >> %log_path%\result.log
        echo   predict of seq2seq_base failed!
) else (
        move  %log_path%\seq2seq_base_I.log  %log_path%\SUCCESS\seq2seq_base_I.log
        echo   seq2seq_base,predict,SUCCESS  >> %log_path%\result.log
        echo   predict of seq2seq_base successfully!
)
