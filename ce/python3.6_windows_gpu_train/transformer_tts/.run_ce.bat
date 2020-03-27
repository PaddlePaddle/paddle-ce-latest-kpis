@echo off
set FLAGS_fraction_of_gpu_memory_to_use=0.5
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fast_eager_deletion_mode=1

set CUDA_VISIBLE_DEVICES=0
rem train transformer
python train_transformer.py --use_gpu=1 --use_data_parallel=0 --data_path=data/LJSpeech-1.1 --config_path=configs/train_transformer.yaml --batch_size 16 --epoch 1 > %log_path%/transformer_tts_train.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\transformer_tts_train.log  %log_path%\FAIL\transformer_tts_train.log
        echo   transformer_tts,train,FAIL  >> %log_path%\result.log
        echo   train of transformer_tts failed!
) else (
        move  %log_path%\transformer_tts_train.log  %log_path%\SUCCESS\transformer_tts_train.log
        echo   transformer_tts,train,SUCCESS  >> %log_path%\result.log
        echo   train of transformer_tts successfully!
)
rem train vocoder
python train_vocoder.py --use_gpu=1 --use_data_parallel=0 --data_path=data/LJSpeech-1.1 --config_path=configs/train_transformer.yaml --epochs=1 --batch_size 16 > %log_path%/transformer_vocoder_train.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\transformer_vocoder_train.log  %log_path%\FAIL\transformer_vocoder_train.log
        echo   transformer_vocoder,train,FAIL  >> %log_path%\result.log
        echo   train of transformer_vocoder failed!
) else (
        move  %log_path%\transformer_vocoder_train.log  %log_path%\SUCCESS\transformer_vocoder_train.log
        echo   transformer_vocoder,train,SUCCESS  >> %log_path%\result.log
        echo   train of transformer_vocoder successfully!
)
rem synthesis
python synthesis.py --max_len=50 --transformer_step=500 --vocoder_step=500 --use_gpu=1 --checkpoint_path=./checkpoint --sample_path=./sample  --config_path=configs/synthesis.yaml > %log_path%/transformer_tts_synthesis.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\transformer_tts_synthesis.log  %log_path%\FAIL\transformer_tts_synthesis.log
        echo   transformer_tts,synthesis,FAIL  >> %log_path%\result.log
        echo   synthesis of transformer_tts failed!
) else (
        move  %log_path%\transformer_tts_synthesis.log  %log_path%\SUCCESS\transformer_tts_synthesis.log
        echo   transformer_tts,synthesis,SUCCESS  >> %log_path%\result.log
        echo   synthesis of transformer_tts successfully!
)


