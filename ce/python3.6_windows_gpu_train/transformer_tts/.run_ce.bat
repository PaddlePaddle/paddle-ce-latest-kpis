@echo off
set FLAGS_fraction_of_gpu_memory_to_use=0.5
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fast_eager_deletion_mode=1

set CUDA_VISIBLE_DEVICES=0
rem train transformer
if exist experiment (rd /s /q experiment)
if exist vocoder (rd /s /q vocoder)

set sed="C:\Program Files\Git\usr\bin\sed.exe"
%sed% -i s/"  max_epochs: 10000"/"  max_epochs: 1"/g configs/ljspeech.yaml
%sed% -i s/"  image_interval: 2000"/"  image_interval: 800"/g configs/ljspeech.yaml
%sed% -i s/"  checkpoint_interval: 1000"/"  checkpoint_interval: 800"/g configs/ljspeech.yaml
%sed% -i s/"  batch_size: 32"/"  batch_size: 16"/g configs/ljspeech.yaml

rem train transformer
python train_transformer.py --use_gpu=1 --data=data/LJSpeech-1.1/ --output=./experiment --config=configs/ljspeech.yaml  > %log_path%/transformer_tts_train.log 2>&1
if %errorlevel% GTR 0 (
        move  %log_path%\transformer_tts_train.log  %log_path%\FAIL\transformer_tts_train.log
        echo   transformer_tts,train,FAIL  >> %log_path%\result.log
        echo   train of transformer_tts failed!
) else (
        move  %log_path%\transformer_tts_train.log  %log_path%\SUCCESS\transformer_tts_train.log
        echo   transformer_tts,train,SUCCESS  >> %log_path%\result.log
        echo   train of transformer_tts successfully!
)
rem train vocoder
python train_vocoder.py --use_gpu=1 --data=data/LJSpeech-1.1/ --output=./vocoder --config=configs/ljspeech.yaml > %log_path%/transformer_vocoder_train.log 2>&1
if %errorlevel% GTR 0 (
        move  %log_path%\transformer_vocoder_train.log  %log_path%\FAIL\transformer_vocoder_train.log
        echo   transformer_vocoder,train,FAIL  >> %log_path%\result.log
        echo   train of transformer_vocoder failed!
) else (
        move  %log_path%\transformer_vocoder_train.log  %log_path%\SUCCESS\transformer_vocoder_train.log
        echo   transformer_vocoder,train,SUCCESS  >> %log_path%\result.log
        echo   train of transformer_vocoder successfully!
)
rem synthesis
python synthesis.py --max_len=300  --use_gpu=1 --output=./synthesis --config=configs/ljspeech.yaml --checkpoint_transformer=./experiment/checkpoints/step-800 --checkpoint_vocoder=./vocoder/checkpoints/step-800 > %log_path%/transformer_tts_synthesis.log 2>&1
if %errorlevel% GTR 0 (
        move  %log_path%\transformer_tts_synthesis.log  %log_path%\FAIL\transformer_tts_synthesis.log
        echo   transformer_tts,synthesis,FAIL  >> %log_path%\result.log
        echo   synthesis of transformer_tts failed!
) else (
        move  %log_path%\transformer_tts_synthesis.log  %log_path%\SUCCESS\transformer_tts_synthesis.log
        echo   transformer_tts,synthesis,SUCCESS  >> %log_path%\result.log
        echo   synthesis of transformer_tts successfully!
)


