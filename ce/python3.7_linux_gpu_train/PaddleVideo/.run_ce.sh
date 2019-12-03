#!/bin/bash

export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98

export CUDA_VISIBLE_DEVICES=0
sed -i "s/num_gpus: 8/num_gpus: 1/g" configs/tsm.yaml
python train.py --model_name="TSM" --config=./configs/tsm.yaml --epoch=1 --log_interval=10 --batch_size=16 --fix_random_seed=True 1> log_1card
cat log_1card | python _ce.py

export CUDA_VISIBLE_DEVICES=0,1,2,3
sed -i "s/num_gpus: 1/num_gpus: 4/g" configs/tsm.yaml
python train.py --model_name="TSM" --config=./configs/tsm.yaml --epoch=1 --log_interval=10 --batch_size=16 --fix_random_seed=True 1> log_4cards
cat log_4cards | python _ce.py


