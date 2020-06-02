@echo off
set FLAGS_fraction_of_gpu_memory_to_use=0.5
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fast_eager_deletion_mode=1

set CUDA_VISIBLE_DEVICES=0
set sed="C:\Program Files\Git\usr\bin\sed.exe"
%sed% -i s/"  max_epochs: 10000"/"  max_epochs: 1"/g configs/ljspeech.yaml
%sed% -i s/"  checkpoint_interval: 1000"/"  checkpoint_interval: 800"/g configs/ljspeech.yaml
%sed% -i s/"  batch_size: 32"/"  batch_size: 16"/g configs/ljspeech.yaml

rd /s /q alignments
mklink /j alignments %data_path%\Parakeet\fastspeech\alignments
if exist experiment (rd /s /q experiment)
python -u train.py --use_gpu=1 --data=data/LJSpeech-1.1 --alignments_path=./alignments/alignments.txt --output=./experiment --config=configs/ljspeech.yaml > %log_path%/fastspeech_train.log
if not %errorlevel% == 0 (
        move  %log_path%\fastspeech_train.log  %log_path%\FAIL\fastspeech_train.log
        echo   fastspeech,train,FAIL  >> %log_path%\result.log
        echo   train of fastspeech failed!
) else (
        move  %log_path%\fastspeech_train.log  %log_path%\SUCCESS\fastspeech_train.log
        echo   fastspeech,train,SUCCESS  >> %log_path%\result.log
        echo   train of fastspeech successfully!
)
if not exist clarinet (mklink /j clarinet %data_path%\Parakeet\clarinet)
if not exist fastspeech (mklink /j fastspeech %data_path%\Parakeet\fastspeech)

python synthesis.py --use_gpu=1 --alpha=1.0 --checkpoint=./fastspeech/step-130000 --config=configs/ljspeech.yaml --config_clarine=./clarinet/config.yaml --checkpoint_clarinet=./clarinet/step-500000  --output=./synthesis > %log_path%/fastspeech_synthesis.log
if not %errorlevel% == 0 (
        move  %log_path%\fastspeech_synthesis.log  %log_path%\FAIL\fastspeech_synthesis.log
        echo   fastspeech,synthesis,FAIL  >> %log_path%\result.log
        echo   synthesis of fastspeech failed!
) else (
        move  %log_path%\fastspeech_synthesis.log  %log_path%\SUCCESS\fastspeech_synthesis.log
        echo   fastspeech,synthesis,SUCCESS  >> %log_path%\result.log
        echo   synthesis of fastspeech successfully!
)

