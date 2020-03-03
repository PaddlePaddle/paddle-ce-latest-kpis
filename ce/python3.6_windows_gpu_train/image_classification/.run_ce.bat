@echo off
rem This file is only used for continuous evaluation.
set FLAGS_cudnn_deterministic=True
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fast_eager_deletion_mode=1
set FLAGS_fraction_of_gpu_memory_to_use=0

set CUDA_VISIBLE_DEVICES=0
set PATH=C:\Program Files (x86)\GnuWin32\bin;%PATH%

setlocal enabledelayedexpansion
for %%I in (AlexNet DPN107 DarkNet53 DenseNet121 EfficientNet HRNet_W18_C GoogLeNet InceptionV4 Xception65_deeplab MobileNetV1 MobileNetV2 ResNet50 ResNet152_vd Res2Net50_vd_26w_4s ResNeXt101_32x4d ResNeXt101_32x8d_wsl SE_ResNeXt50_vd_32x4d ShuffleNetV2_swish SqueezeNet1_1 VGG19) do (
python train.py --model=%%I --num_epochs=1 --batch_size 8 --lr_strategy=cosine_decay --random_seed 1000 --use_gpu true --enable_ce=True > %%I.log 2>&1
type %%I.log | grep "train_cost_card1" | gawk "{print \"kpis\t\"\"%%I\"\"_loss_card1\t\"$5}" | python _ce.py
type %%I.log | grep "train_speed_card1" | gawk "{print \"kpis\t\"\"%%I\"\"_time_card1\t\"$5}" | python _ce.py
rem eval
python eval.py --model=%%I --batch_size=32 --pretrained_model=output/%%I/0 --use_gpu true >%log_path%/%%I_E.log 2>&1
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
python infer.py --model=%%I --pretrained_model=output/%%I/0 --use_gpu True --data_dir=data/ILSVRC2012/test >%log_path%/%%I_I.log 2>&1
if not !errorlevel! == 0 (
        move  %log_path%\%%I_I.log  %log_path%\FAIL\%%I_I.log
        echo   %%I,infer,FAIL  >> %log_path%\result.log
        echo  infering of %%I failed!
) else (
        move  %log_path%\%%I_I.log  %log_path%\SUCCESS\%%I_I.log
        echo   %%I,infer,SUCCESS  >> %log_path%\result.log
        echo   infering of %%I successfully!
)
rem save_inference
python infer.py  --model=%%I  --pretrained_model=output/%%I/0 --save_inference=True >%log_path%/%%I_SI.log 2>&1
if not !errorlevel! == 0 (
        move  %log_path%\%%I_SI.log  %log_path%\FAIL\%%I_SI.log
        echo   %%I,save_inference,FAIL  >> %log_path%\result.log
        echo  save_inference of %%I failed!
) else (
        move  %log_path%\%%I_SI.log  %log_path%\SUCCESS\%%I_SI.log
        echo   %%I,save_inference,SUCCESS  >> %log_path%\result.log
        echo   save_inference of %%I successfully!
)
rem predict
python predict.py  --model_file=%%I/model --params_file=%%I/params  --image_path=data/ILSVRC2012/test/ILSVRC2012_val_00000001.jpeg  --data_dir=data/ILSVRC2012/test --gpu_id=0  --gpu_mem=1024 >%log_path%/%%I_P.log 2>&1
if not !errorlevel! == 0 (
        move  %log_path%\%%I_P.log  %log_path%\FAIL\%%I_P.log
        echo   %%I,predict,FAIL  >> %log_path%\result.log
        echo  predict of %%I failed!
) else (
        move  %log_path%\%%I_P.log  %log_path%\SUCCESS\%%I_P.log
        echo   %%I,predict,SUCCESS  >> %log_path%\result.log
        echo   predict of %%I successfully!
)
)
