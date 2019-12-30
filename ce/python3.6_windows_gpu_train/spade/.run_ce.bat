@echo off
rem This file is only used for continuous evaluation.
set FLAGS_cudnn_deterministic=True
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fast_eager_deletion_mode=1
set FLAGS_fraction_of_gpu_memory_to_use=0

set CUDA_VISIBLE_DEVICES=0

python train.py --model_net SPADE --output output_spade  --dataset cityscapes --train_list ./data/cityscapes/train_list --test_list ./data/cityscapes/val_list --batch_size 1 --epoch 1 --load_height 612 --load_width 1124 --crop_height 512 --crop_width 1024 --label_nc 36 --use_gpu true --shuffle false --print_freq 1 --save_checkpoints false --enable_ce true --run_test true | python _ce.py

