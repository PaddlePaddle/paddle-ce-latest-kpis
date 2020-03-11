#!/bin/bash
# This file is only used for continuous evaluation.
export FLAGS_cudnn_deterministic=True
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_fraction_of_gpu_memory_to_use=0
export CPU_NUM=12
python train.py  --model_net DCGAN --output output_dcgan --dataset mnist --shuffle False --batch_size 1200 --epoch 1 --enable_ce --use_gpu False --print_freq 1 | python _ce.py

