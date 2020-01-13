@echo off
rem This file is only used for continuous evaluation.
set FLAGS_cudnn_deterministic=True
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fast_eager_deletion_mode=1
set FLAGS_fraction_of_gpu_memory_to_use=0

set CUDA_VISIBLE_DEVICES=0
python train.py --model_net AttGAN --output output_attgan --dataset celeba --crop_size 170 --image_size 128 --train_list ./data/celeba/list_attr_celeba.txt --test_list ./data/celeba/test_list_attr_celeba.txt  --gan_mode wgan  --batch_size 32 --print_freq 1 --num_discriminator_time 5 --epoch 1  --run_test false --save_checkpoints false --shuffle false --use_gpu true --num_discriminator_time 1 --enable_ce | python _ce.py

