@echo off
rem This file is only used for continuous evaluation.

set CPU_NUM=1
rem train
python train.py --model_net CycleGAN --dataset cityscapes  --net_G resnet_9block --g_base_dim 32 --net_D basic --norm_type batch_norm --image_size 286 --crop_size 256 --output ./output/cyclegan/  --epoch 1 --enable_ce --shuffle False --run_test False --save_checkpoints True --use_gpu False  | python _ce.py
rem infer 
python infer.py --init_model output/cyclegan/checkpoints/0/ --dataset_dir "data/cityscapes/testB/*" --image_size 256 --input_style B --model_net CycleGAN --net_G resnet_9block  --g_base_dims 32 --use_gpu false > %log_path%/cyclgan_I.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\cyclegan_I.log  %log_path%\FAIL\cyclegan_I.log
        echo   cycle_gan,infer,FAIL  >> %log_path%\result.log
        echo  infer of cycle_gan failed!
) else (
        move  %log_path%\cyclegan_I.log  %log_path%\SUCCESS\cyclegan_I.log
        echo   cycle_gan,infer,SUCCESS  >> %log_path%\result.log
        echo   infer of cycle_gan successfully!
)
