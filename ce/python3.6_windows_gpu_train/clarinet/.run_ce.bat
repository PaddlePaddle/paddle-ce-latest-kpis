@echo off
set FLAGS_fraction_of_gpu_memory_to_use=0.5
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fast_eager_deletion_mode=1

set CUDA_VISIBLE_DEVICES=0
set sed="C:\Program Files\Git\usr\bin\sed.exe"
 

%sed% -i s/"  max_iterations: 2000000"/"  max_iterations: 6"/g configs/clarinet_ljspeech.yaml
%sed% -i s/"  checkpoint_interval: 1000"/"  checkpoint_interval: 3"/g configs/clarinet_ljspeech.yaml
%sed% -i s/"  eval_interval: 1000"/"  eval_interval: 3"/g configs/clarinet_ljspeech.yaml
%sed% -i s/"  batch_size: 8"/"  batch_size: 1"/g configs/clarinet_ljspeech.yaml
%sed% -i s/"  valid_size: 16"/"  valid_size: 1"/g configs/clarinet_ljspeech.yaml
if exist experiment (rd /s /q experiment)
if not exist wavenet (mklink /j wavenet %data_path%\Parakeet\wavenet)
python train.py --config=./configs/clarinet_ljspeech.yaml --data=data/LJSpeech-1.1/ --device=0  --wavenet="./wavenet/step-2" experiment > %log_path%/clarinet_train.log
if not %errorlevel% == 0 (
        move  %log_path%\clarinet_train.log  %log_path%\FAIL\clarinet_train.log
        echo   clarinet,train,FAIL  >> %log_path%\result.log
        echo   train of clarinet failed!
) else (
        move  %log_path%\clarinet_train.log  %log_path%\SUCCESS\clarinet_train.log
        echo   clarinet,train,SUCCESS  >> %log_path%\result.log
        echo   train of clarinet successfully!
)
python synthesis.py --config=./configs/clarinet_ljspeech.yaml --data=data/LJSpeech-1.1/ --device=0  --checkpoint="experiment/checkpoints/step-6" experiment > %log_path%/clarinet_synthesis.log
if not %errorlevel% == 0 (
        move  %log_path%\clarinet_synthesis.log  %log_path%\FAIL\clarinet_synthesis.log
        echo   clarinet,synthesis,FAIL  >> %log_path%\result.log
        echo   synthesis of clarinet failed!
) else (
        move  %log_path%\clarinet_synthesis.log  %log_path%\SUCCESS\clarinet_synthesis.log
        echo   clarinet,synthesis,SUCCESS  >> %log_path%\result.log
        echo   synthesis of clarinet successfully!
)

