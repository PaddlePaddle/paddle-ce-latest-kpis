@echo off
rem This file is only used for continuous evaluation.
set FLAGS_cudnn_deterministic=True
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fast_eager_deletion_mode=1
set FLAGS_fraction_of_gpu_memory_to_use=0

set CUDA_VISIBLE_DEVICES=0
python train.py --model_net Pix2pix --output output_pix2pix --net_G unet_256  --dataset cityscapes --train_list data/cityscapes/pix2pix_train_list --test_list data/cityscapes/pix2pix_test_list  --dropout False --gan_mode vanilla --batch_size 64 --epoch 1 --enable_ce --shuffle False --run_test False --save_checkpoints False --use_gpu True  | python _ce.py

