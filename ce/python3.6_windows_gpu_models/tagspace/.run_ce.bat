@echo off
set CUDA_VISIBLE_DEVICES=0
rem train
python train.py --use_cuda 1 --train_dir train_data/ --vocab_text_path vocab_text.txt --vocab_tag_path vocab_tag.txt --model_dir output --batch_size 500 --pass_num 10 > %log_path%\tagsapce_T.log 2>&1
 if not %errorlevel% == 0 (
        move  %log_path%\tagspace_T.log  %log_path%\FAIL\tagspace_T.log
        echo  tagspace_T,train,FAIL  >> %log_path%\result.log
        echo  training of tagspace_T failed!
) else (
        move  %log_path%\tagspace_T.log  %log_path%\SUCCESS\tagspace_T.log
        echo   tagspace_T,train,SUCCESS  >> %log_path%\result.log
        echo   training of tagspace_T successfully!
 )

rem infer
python infer.py --use_cuda 1 --model_dir output --vocab_tag_path vocab_tag.txt --test_dir test_data --start_index 1 --last_index 10 > %log_path%\tagsapce_I.log 2>&1
type tagspace_I.log |grep acc |gawk -F ":" "{print $3}"|gawk "NR==1{max=$1;next}{max=max>$1?max:$1}END{print  \"kpis\ttest_acc\t\"max}"| python _ce.py
if not %errorlevel% == 0 (
        move  %log_path%\tagspace_I.log  %log_path%\FAIL\tagsapce_I.log
        echo   tagspace,infer,FAIL  >> %log_path%\result.log
        echo    infer of tagspace failed!
) else (
        move  %log_path%\tagsapce_I.log  %log_path%\SUCCESS\tagsapce_I.log
        echo  tagspace,infer,SUCCESS  >> %log_path%\result.log
        echo  infer of tagspace successfully!
)
