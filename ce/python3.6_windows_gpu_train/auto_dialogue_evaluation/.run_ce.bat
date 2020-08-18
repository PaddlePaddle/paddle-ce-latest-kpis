@echo off 
set FLAGS_sync_nccl_allreduce=0
set FLAGS_eager_delete_tensor_gb=1.0

set CUDA_VISIBLE_DEVICES=0	
set ce_mode=1  
rem train
python -u main.py --do_train=true --use_cuda=true --loss_type="CLS"  --max_seq_len=50  --save_model_path="data/saved_models/matching_pretrained"  --save_param="params"  --training_file="data/input/data/unlabel_data/train.ids" --epoch=20 --print_step=1 --save_step=400 --batch_size=256  --hidden_size=256 --emb_size=256  --vocab_size=484016 --learning_rate=0.001 --sample_pro=0.1 --enable_ce="store_true" | python _ce.py
rem infer
python -u main.py --do_predict=true --use_cuda=true --predict_file="data/input/data/unlabel_data/test.ids" --init_from_params="data/saved_models/trained_models/matching_pretrained/params" --loss_type="CLS" --output_prediction_file="data/output/pretrain_matching_predict" --max_seq_len=50 --batch_size=256 --hidden_size=256  --emb_size=256 --vocab_size=484016 > %log_path%/dialogue_evaluation_I.log 2>&1
if %errorlevel% GTR 0 (
        move  %log_path%\dialogue_evaluation_I.log  %log_path%\FAIL\dialogue_evaluation_I.log
        echo   dialogue_evaluation,infer,FAIL  >> %log_path%\result.log
        echo   infer of dialogue_evaluation failed!
) else (
        move  %log_path%\dialogue_evaluation_I.log  %log_path%\SUCCESS\dialogue_evaluation_I.log
        echo   dialogue_evaluation,infer,SUCCESS  >> %log_path%\result.log
        echo   infer of dialogue_evaluation successfully!
 )




