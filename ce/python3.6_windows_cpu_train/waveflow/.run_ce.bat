@echo off
set FLAGS_fraction_of_gpu_memory_to_use=0.5
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fast_eager_deletion_mode=1

set CUDA_VISIBLE_DEVICES=0
set sed="C:\Program Files\Git\usr\bin\sed.exe"
 

%sed% -i s/"max_iterations: 3000000"/"max_iterations: 2"/g configs/waveflow_ljspeech.yaml
%sed% -i s/"save_every: 10000"/"save_every: 1"/g configs/waveflow_ljspeech.yaml
%sed% -i s/"test_every: 2000"/"test_every: 1"/g configs/waveflow_ljspeech.yaml
if exist runs (rd /s /q runs)
python -u train.py --config=./configs/waveflow_ljspeech.yaml --root=data/LJSpeech-1.1  --name=waveflow  --batch_size=4 --use_gpu=false > %log_path%/waveflow_train.log
if not %errorlevel% == 0 (
        move  %log_path%\waveflow_train.log  %log_path%\FAIL\waveflow_train.log
        echo   waveflow,train,FAIL  >> %log_path%\result.log
        echo   train of waveflow failed!
) else (
        move  %log_path%\waveflow_train.log  %log_path%\SUCCESS\waveflow_train.log
        echo   waveflow,train,SUCCESS  >> %log_path%\result.log
        echo   train of waveflow successfully!
)
python -u synthesis.py --config=./configs/waveflow_ljspeech.yaml --root=data/LJSpeech-1.1 --name=waveflow --use_gpu=false --output=./syn_audios --sample=0 --sigma=1.0 > %log_path%/waveflow_synthesis.log
if not %errorlevel% == 0 (
        move  %log_path%\waveflow_synthesis.log  %log_path%\FAIL\waveflow_synthesis.log
        echo   waveflow,synthesis,FAIL  >> %log_path%\result.log
        echo   synthesis of waveflow failed!
) else (
        move  %log_path%\waveflow_synthesis.log  %log_path%\SUCCESS\waveflow_synthesis.log
        echo   waveflow,synthesis,SUCCESS  >> %log_path%\result.log
        echo   synthesis of waveflow successfully!
)

