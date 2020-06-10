@echo off
set FLAGS_fraction_of_gpu_memory_to_use=0.5
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fast_eager_deletion_mode=1

set CUDA_VISIBLE_DEVICES=0
set sed="C:\Program Files\Git\usr\bin\sed.exe"
 
%sed% -i s/"  max_iterations: 2000000"/"  max_iterations: 2"/g configs/wavenet_single_gaussian.yaml
%sed% -i s/"  checkpoint_interval: 10000"/"  checkpoint_interval: 1"/g configs/wavenet_single_gaussian.yaml
%sed% -i s/"  snap_interval: 10000"/"  snap_interval: 1"/g configs/wavenet_single_gaussian.yaml
%sed% -i s/"  eval_interval: 10000"/"  eval_interval: 1"/g configs/wavenet_single_gaussian.yaml

if exist experiment (rd /s /q experiment)

python train.py --config=./configs/wavenet_single_gaussian.yaml --data=data/LJSpeech-1.1/  --device=-1 experiment > %log_path%/wavenet_train.log
if not %errorlevel% == 0 (
        move  %log_path%\wavenet_train.log  %log_path%\FAIL\wavenet_train.log
        echo   wavenet,train,FAIL  >> %log_path%\result.log
        echo   train of wavenet failed!
) else (
        move  %log_path%\wavenet_train.log  %log_path%\SUCCESS\wavenet_train.log
        echo   wavenet,train,SUCCESS  >> %log_path%\result.log
        echo   train of wavenet successfully!
)
python synthesis.py --config=./configs/wavenet_single_gaussian.yaml  --data=data/LJSpeech-1.1/ --device=-1 --checkpoint="experiment/checkpoints/step-2" experiment > %log_path%/wavenet_synthesis.log
if not %errorlevel% == 0 (
        move  %log_path%\wavenet_synthesis.log  %log_path%\FAIL\wavenet_synthesis.log
        echo   wavenet,synthesis,FAIL  >> %log_path%\result.log
        echo   synthesis of wavenet failed!
) else (
        move  %log_path%\wavenet_synthesis.log  %log_path%\SUCCESS\wavenet_synthesis.log
        echo   wavenet,synthesis,SUCCESS  >> %log_path%\result.log
        echo   synthesis of wavenet successfully!
)

