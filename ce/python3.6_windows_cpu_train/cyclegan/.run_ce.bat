@echo off
rem This file is only used for continuous evaluation.

set CPU_NUM=1
python train.py --model_net CycleGAN --dataset cityscapes  --net_G resnet_9block --g_base_dim 32 --net_D basic --norm_type batch_norm --image_size 286 --crop_size 256 --output ./output/cyclegan/  --epoch 1 --enable_ce --shuffle False --run_test True --save_checkpoints True --use_gpu False  | python _ce.py

