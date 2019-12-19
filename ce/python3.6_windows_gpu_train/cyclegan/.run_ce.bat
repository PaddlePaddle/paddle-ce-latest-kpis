@echo off
rem This file is only used for continuous evaluation.
set FLAGS_cudnn_deterministic=True
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fast_eager_deletion_mode=1
set FLAGS_fraction_of_gpu_memory_to_use=0

set CUDA_VISIBLE_DEVICES=0
python train.py --model_net CycleGAN --dataset cityscapes  --net_G resnet_9block --g_base_dim 32 --net_D basic --norm_type batch_norm --image_size 286 --crop_size 256 --output ./output/cyclegan/  --epoch 1 --enable_ce --shuffle False --run_test True --save_checkpoints True --use_gpu True  | python _ce.py

