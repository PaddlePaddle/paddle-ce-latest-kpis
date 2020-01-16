@echo off
rem This file is only used for continuous evaluation.
set FLAGS_cudnn_deterministic=True
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fast_eager_deletion_mode=1
set FLAGS_fraction_of_gpu_memory_to_use=0

setlocal enabledelayedexpansion
for %%I in (AlexNet DPN107 DarkNet53 DenseNet121 EfficientNet HRNet_W18_C GoogLeNet InceptionV4 Xception65_deeplab MobileNetV1 MobileNetV2 ResNet50 ResNet152_vd Res2Net50_vd_26w_4s ResNeXt101_32x4d ResNeXt101_32x8d_wsl SE_ResNeXt50_vd_32x4d ShuffleNetV2_swish SqueezeNet1_1 VGG19) do (
python train.py --model=%%I --num_epochs=1 --batch_size 8 --lr=0.01 --lr_strategy=cosine_decay --random_seed 1000 --use_gpu false --enable_ce=True > %%I.log 2>&1
type %%I.log | grep "train_cost_card1" | gawk "{print \"kpis\t\"\"%%I\"\"_loss_card1\t\"$5}" | python _ce.py
type %%I.log | grep "train_speed_card1" | gawk "{print \"kpis\t\"\"%%I\"\"_time_card1\t\"$5}" | python _ce.py
rem eval
python eval.py --model=%%I --batch_size=32 --pretrained_model=output/%%I/0 --use_gpu false >%log_path%/%%I_E.log 2>&1
if not !errorlevel! == 0 (
        move  %log_path%\%%I_E.log  %log_path%\FAIL\%%I_E.log
        echo   %%I,eval,FAIL  >> %log_path%\result.log
        echo  evaling of %%I failed!
) else (
        move  %log_path%\%%I_E.log  %log_path%\SUCCESS\%%I_E.log
        echo   %%I,eval,SUCCESS  >> %log_path%\result.log
        echo   evaling of %%I successfully!
)
rem infer
python infer.py --model=%%I --pretrained_model=output/%%I/0 --use_gpu false >%log_path%/%%I_I.log 2>&1
if not !errorlevel! == 0 (
        move  %log_path%\%%I_I.log  %log_path%\FAIL\%%I_I.log
        echo   %%I,infer,FAIL  >> %log_path%\result.log
        echo  infering of %%I failed!
) else (
        move  %log_path%\%%I_I.log  %log_path%\SUCCESS\%%I_I.log
        echo   %%I,infer,SUCCESS  >> %log_path%\result.log
        echo   infering of %%I successfully!
)
)
