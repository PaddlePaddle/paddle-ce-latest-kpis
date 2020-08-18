@echo off
set CUDA_VISIBLE_DEVICES=0
rem train
python train_elem.py --model=ResNet50 --loss_name arcmargin --total_iter_num=10 --test_iter_step=10 --model_save_dir=output --save_iter_step=10 --train_batch_size 16 --test_batch_siz 16 --use_gpu=True --enable_ce true | python _ce.py 
rem eval
python eval.py --model=ResNet50 --batch_size=16 --pretrained_model=output --use_gpu True > %log_path%/metric_E.log 2>&1
if %errorlevel% GTR 0 (
        move  %log_path%\metric_E.log  %log_path%\FAIL\metric_E.log
        echo   metric,eval,FAIL  >> %log_path%\result.log
        echo  eval of metric_learning failed!
 ) else (
        move  %log_path%\metric_E.log  %log_path%\SUCCESS\metric_E.log
        echo   metric,eval,SUCCESS  >> %log_path%\result.log
        echo   eval of metric_learning successfully!
 )
rem infer
python infer.py --model=ResNet50 --batch_size=16 --pretrained_model=output  --use_gpu True> %log_path%/metric_I.log 2>&1
if %errorlevel% GTR 0 (
        move  %log_path%\metric_I.log  %log_path%\FAIL\metric_I.log
        echo  metric,infer,FAIL  >> %log_path%\result.log
        echo  infer of metric_learning failed!
 ) else (
        move  %log_path%\metric_I.log  %log_path%\SUCCESS\metric_I.log
        echo   metric,infer,SUCCESS  >> %log_path%\result.log
        echo  infer of metric_learning successfully!
 )