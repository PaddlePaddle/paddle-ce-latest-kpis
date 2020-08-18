@echo off
set FLAGS_fraction_of_gpu_memory_to_use=0.5
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fast_eager_deletion_mode=1
set CUDA_VISIBLE_DEVICES=0
rem train
python -u train.py --epoch 1 --src_vocab_fpath data/wmt16_ende_data_bpe_clean/vocab_all.bpe.32000 --trg_vocab_fpath data/wmt16_ende_data_bpe_clean/vocab_all.bpe.32000 --training_file data/wmt16_ende_data_bpe_clean/newstest2016.tok.bpe.32000.en-de --validation_file data/wmt16_ende_data_bpe_clean/newstest2016.tok.bpe.32000.en-de --batch_size 1024 --print_step 1 
type  hapi_transformer.log|grep "step 91/91"|gawk -F "[:-]" "NR==1{print \"kpis\ttrain_loss\t\"$3}" | python _ce.py
type  hapi_transformer.log|grep "step 91/91"|gawk -F "[:-]" "END{print \"kpis\teval_loss\t\"$3}" | python _ce.py

rem predict
python -u predict.py  --src_vocab_fpath data/wmt16_ende_data_bpe_clean/vocab_all.bpe.32000   --trg_vocab_fpath data/wmt16_ende_data_bpe_clean/vocab_all.bpe.32000  --predict_file data/wmt16_ende_data_bpe_clean/newstest2016.tok.bpe.32000.en-de  --batch_size 32 --init_from_params trained_models/0  --beam_size 5 --max_out_len 255 --output_file predict.txt > %log_path%\hapi_tranformer_I.log 
if %errorlevel% GTR 0 (
        move  %log_path%\hapi_tranformer_I.log  %log_path%\FAIL\hapi_tranformer_I.log
        echo   hapi_tranformer,infer,FAIL  >> %log_path%\result.log
        echo  infering of hapi_tranformer failed!
) else (
        move  %log_path%\hapi_tranformer_I.log  %log_path%\SUCCESS\hapi_tranformer_I.log
        echo   hapi_tranformer,infer,SUCCESS  >> %log_path%\result.log
        echo   infering of hapi_tranformer successfully!
)


