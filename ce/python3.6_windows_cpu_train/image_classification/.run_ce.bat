@echo off
rem This file is only used for continuous evaluation.
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fast_eager_deletion_mode=1
set FLAGS_fraction_of_gpu_memory_to_use=0
rem train
python train.py --model=ResNet50 --num_epochs=2 --batch_size 8 --lr_strategy=cosine_decay --pretrained_model=ResNet50_pretrained --random_seed 1000 --use_gpu false --enable_ce=True 2>log
type log  | python _ce.py

rem eval
python eval.py --model=ResNet50 --batch_size=32 --pretrained_model=ResNet50_pretrained --use_gpu False >%log_path%/ResNet50_E.log 2>&1 
if not %errorlevel% == 0 (
        move  %log_path%\ResNet50_E.log  %log_path%\FAIL\ResNet50_E.log
        echo   ResNet50,eval,FAIL  >> %log_path%\result.log
        echo  evaling of ResNet50 failed!
) else (
        move  %log_path%\ResNet50_E.log  %log_path%\SUCCESS\ResNet50_E.log
        echo   ResNet50,eval,SUCCESS  >> %log_path%\result.log
        echo   evaling of ResNet50 successfully!
)
rem infer
python infer.py --model=ResNet50 --pretrained_model=ResNet50_pretrained --use_gpu False >%log_path%/ResNet50_I.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\ResNet50_I.log  %log_path%\FAIL\ResNet50_I.log
        echo   ResNet50,infer,FAIL  >> %log_path%\result.log
        echo  infering of ResNet50 failed!
) else (
        move  %log_path%\ResNet50_I.log  %log_path%\SUCCESS\ResNet50_I.log
        echo   ResNet50,infer,SUCCESS  >> %log_path%\result.log
        echo   infering of ResNet50 successfully!
)

