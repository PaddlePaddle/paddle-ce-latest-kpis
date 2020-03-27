@echo off
set FLAGS_fraction_of_gpu_memory_to_use=0.5
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fast_eager_deletion_mode=1
pip install -U paddlehub
set CUDA_VISIBLE_DEVICES=0
rem train
python train.py --train_data ./data/train.tsv --test_data ./data/test.tsv --model_save_dir ./models --validation_steps 2 --save_steps 1000 --print_steps 1 --batch_size 300 --epoch 100 --use_cuda true --traindata_shuffle_buffer 200000 --word_emb_dim 128 --grnn_hidden_dim 128 --bigru_num 2  --base_learning_rate 1e-3 --emb_learning_rate 2 --crf_learning_rate 0.2 --word_dict_path ./conf/word.dic --label_dict_path ./conf/tag.dic  --word_rep_dict_path ./conf/q2b.dic > %log_path%/lexical_analysis_T.log  2>&1 
| python _ce.py				
type %log_path%\lexical_analysis_T.log |grep "\[test\]"| awk  -F "[:,]" "{print $2}" | gawk "NR==1{max=$1;next}{max=max>$1?max:$1}END{print  \"kpis\ttest_precision\t\"max}" | python _ce.py
type %log_path%\lexical_analysis_T.log |grep "\[test\]"| awk  -F "[:,]" "{print $2}" | gawk "NR==1{max=$1;next}{max=max>$1?max:$1}END{print  \"kpis\ttest_recall\t\"max}" | python _ce.py
type %log_path%\lexical_analysis_T.log |grep "\[test\]"| awk  -F "[:,]" "{print $2}" | gawk "NR==1{max=$1;next}{max=max>$1?max:$1}END{print  \"kpis\ttest_F1\t\"max}" | python _ce.py
if not %errorlevel% == 0 (
        move  %log_path%\lexical_analysis_T.log  %log_path%\FAIL\lexical_analysis_T.log
        echo   lac,train,FAIL  >> %log_path%\result.log
        echo   train of lac failed!
) else (
        move  %log_path%\lexical_analysis_T.log  %log_path%\SUCCESS\lexical_analysis_T.log
        echo   lac,train,SUCCESS  >> %log_path%\result.log
        echo   train of lac successfully!
)



