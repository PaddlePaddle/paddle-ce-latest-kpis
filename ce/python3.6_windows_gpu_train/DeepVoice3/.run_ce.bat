@echo off
set FLAGS_fraction_of_gpu_memory_to_use=0.5
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fast_eager_deletion_mode=1

set CUDA_VISIBLE_DEVICES=0
set sed="C:\Program Files\Git\usr\bin\sed.exe"
%sed% -i s/"  epochs: 2000"/"  epochs: 1"/g configs/ljspeech.yaml
%sed% -i s/"  snap_interval: 1000"/"  snap_interval: 816"/g configs/ljspeech.yaml
%sed% -i s/"  eval_interval: 10000"/"  eval_interval: 816"/g configs/ljspeech.yaml
%sed% -i s/"  save_interval: 10000"/"  save_interval: 816"/g configs/ljspeech.yaml

python train.py --config=configs/ljspeech.yaml --data=data/LJSpeech-1.1/ --device=0 experiment > %log_path%/deepvoice3_train.log
if not %errorlevel% == 0 (
        move  %log_path%\deepvoice3_train.log  %log_path%\FAIL\deepvoice3_train.log
        echo   deepvoice3,train,FAIL  >> %log_path%\result.log
        echo   train of deepvoice3 failed!
) else (
        move  %log_path%\deepvoice3_train.log  %log_path%\SUCCESS\deepvoice3_train.log
        echo   deepvoice3,train,SUCCESS  >> %log_path%\result.log
        echo   train of deepvoice3 successfully!
)
python synthesis.py --config=configs/ljspeech.yaml --device=0 --checkpoint=experiment/checkpoints/model_step_005000000 sentences.txt experiment > %log_path%/deepvoice3_synthesis.log
if not %errorlevel% == 0 (
        move  %log_path%\deepvoice3_synthesis.log  %log_path%\FAIL\deepvoice3_synthesis.log
        echo   deepvoice3,synthesis,FAIL  >> %log_path%\result.log
        echo   synthesis of deepvoice3 failed!
) else (
        move  %log_path%\deepvoice3_synthesis.log  %log_path%\SUCCESS\deepvoice3_synthesis.log
        echo   deepvoice3,synthesis,SUCCESS  >> %log_path%\result.log
        echo   synthesis of deepvoice3 successfully!
)

