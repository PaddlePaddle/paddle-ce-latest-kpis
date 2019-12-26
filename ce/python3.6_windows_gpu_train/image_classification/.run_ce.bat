@echo off
rem This file is only used for continuous evaluation.
set FLAGS_cudnn_deterministic=True
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fast_eager_deletion_mode=1
set FLAGS_fraction_of_gpu_memory_to_use=0

set CUDA_VISIBLE_DEVICES=0
python train.py --model=ResNet50 --num_epochs=2 --batch_size 8 --lr_strategy=cosine_decay --pretrained_model=ResNet50_pretrained --random_seed 1000 --use_gpu true --enable_ce=True 2>log
type log  | python _ce.py

