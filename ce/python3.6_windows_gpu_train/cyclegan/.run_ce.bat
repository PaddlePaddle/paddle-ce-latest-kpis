@echo off
rem This file is only used for continuous evaluation.
set FLAGS_cudnn_deterministic=True
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fast_eager_deletion_mode=1
set FLAGS_fraction_of_gpu_memory_to_use=0

set CUDA_VISIBLE_DEVICES=0
python train.py --model_net CycleGAN --dataset cityscapes  --net_G resnet_9block --g_base_dim 32 --net_D basic --norm_type batch_norm --image_size 286 --crop_size 256 --output ./output/cyclegan/  --epoch 1 --enable_ce --shuffle False --run_test True --save_checkpoints True --use_gpu True  | python _ce.py
rem infer 
python infer.py --init_model output/cyclegan/checkpoints/0/ --dataset_dir data/cityscapes/ --image_size 256 --n_samples 20 --crop_size 256 --input_style B --test_list ./data/cityscapes/testB.txt --model_net CycleGAN --net_G resnet_9block --g_base_dims 32 --output ./infer_result/cyclegan/ --use_gpu true > %log_path%/cyclgan_I.log 2>&1
if not %errorlevel% == 0 (
        move  %log_path%\cyclegan_I.log  %log_path%\FAIL\cyclegan_I.log
        echo   cycle_gan,infer,FAIL  >> %log_path%\result.log
        echo  infer of cycle_gan failed!
) else (
        move  %log_path%\cyclegan_I.log  %log_path%\SUCCESS\cyclegan_I.log
        echo   cycle_gan,infer,SUCCESS  >> %log_path%\result.log
        echo   infer of cycle_gan successfully!
)
