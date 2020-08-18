@echo off 
set FLAGS_sync_nccl_allreduce=0
set FLAGS_eager_delete_tensor_gb=1.0

set CUDA_VISIBLE_DEVICES=0
rem atis_intent
rem train
python -u main.py --task_name atis_intent --use_cuda true --do_train=true --epoch=1 --batch_size=32  --do_lower_case=true --data_dir=./data/input/data/atis/atis_intent  --bert_config_path=./data/pretrain_model/uncased_L-12_H-768_A-12/bert_config.json --vocab_path=./data/pretrain_model/uncased_L-12_H-768_A-12/vocab.txt --init_from_pretrain_model=./data/pretrain_model/uncased_L-12_H-768_A-12/params --save_model_path=./data/saved_models/atis_intent --save_param="params" --save_steps=100 --learning_rate=2e-5 --weight_decay=0.01  --max_seq_len=128  --print_steps=10
rem predict
python -u main.py --task_name atis_intent --use_cuda true --do_predict=true --batch_size=32  --do_lower_case=true --data_dir=./data/input/data/atis/atis_intent --bert_config_path=./data/pretrain_model/uncased_L-12_H-768_A-12/bert_config.json --vocab_path=./data/pretrain_model/uncased_L-12_H-768_A-12/vocab.txt --init_from_pretrain_model=./data/pretrain_model/uncased_L-12_H-768_A-12/params --output_prediction_file=./data/output/pred_atis_intent --max_seq_len=128 > %log_path%/dialogue_understanding_predict.log
if %errorlevel% GTR 0 (
        move  %log_path%\dialogue_understanding_predict.log  %log_path%\FAIL\dialogue_understanding_predict.log
        echo   dialogue_understanding,predict,FAIL  >> %log_path%\result.log
        echo   predict of dialogue_understanding failed!
) else (
        move  %log_path%\dialogue_understanding_predict.log  %log_path%\SUCCESS\dialogue_understanding_predict.log
        echo   dialogue_understanding,predict,SUCCESS  >> %log_path%\result.log
        echo   predict of dialogue_understanding successfully!
 )
rem evaluate
python -u main.py --task_name atis_intent --use_cuda true --do_eval=true --evaluation_file=./data/input/data/atis/atis_intent/test.txt --output_prediction_file=./data/output/pred_atis_intent > %log_path%/dialogue_understanding_eval.log
if %errorlevel% GTR 0 (
        move  %log_path%\dialogue_understanding_eval.log  %log_path%\FAIL\dialogue_understanding_eval.log
        echo   dialogue_understanding,eval,FAIL  >> %log_path%\result.log
        echo   eval of dialogue_understanding failed!
) else (
        move  %log_path%\dialogue_understanding_eval.log  %log_path%\SUCCESS\dialogue_understanding_eval.log
        echo   dialogue_understanding,eval,SUCCESS  >> %log_path%\result.log
        echo   eval of dialogue_understanding successfully!
 )
rem save_inference
python -u main.py --task_name atis_intent --use_cuda true --init_from_pretrain_model=./data/pretrain_model/uncased_L-12_H-768_A-12/params --do_save_inference_model=True  --bert_config_path=./data/pretrain_model/uncased_L-12_H-768_A-12/bert_config.json --inference_model_dir=data/inference_models/atis_intent > %log_path%/dialogue_understanding_save_inference.log
if %errorlevel% GTR 0 (
        move  %log_path%\dialogue_understanding_save_inference.log  %log_path%\FAIL\dialogue_understanding_save_inference.log
        echo   dialogue_understanding,save_inference,FAIL  >> %log_path%\result.log
        echo   save_inference of dialogue_understanding failed!
) else (
        move  %log_path%\dialogue_understanding_save_inference.log  %log_path%\SUCCESS\dialogue_understanding_save_inference.log
        echo   dialogue_understanding,save_inference,SUCCESS  >> %log_path%\result.log
        echo   save_inference of dialogue_understanding successfully!
 )
