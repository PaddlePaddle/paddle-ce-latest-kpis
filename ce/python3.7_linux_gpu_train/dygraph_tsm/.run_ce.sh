#!/bin/bash

export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98

export CUDA_VISIBLE_DEVICES=0
python train.py --config=./tsm.yaml --use_gpu=True --epoch=1 --batch_size=16 >log_1card 2>&1
cat log_1card | grep "End" | tail "-1" | tr ',' ' ' | awk '{print "kpis\tloss_card1\t"$6}' | python _ce.py

cat log_1card | grep "End" | tail "-1" | tr ',' ' '| awk '{print "kpis\tacc1_card1\t"$8}' | python _ce.py

export CUDA_VISIBLE_DEVICES=0,1,2,3
sed -i "s/num_gpus: 1/num_gpus: 8/g" ./tsm.yaml
python train.py --config=./tsm.yaml --use_gpu=True --epoch=1 >log_8card 2>&1
cat log_8card | grep "End" | tail "-1" | tr ',' ' ' | awk '{print "kpis\tloss_card8\t"$6}' | python _ce.py

cat log_8card | grep "End" | tail "-1" | tr ',' ' '| awk '{print "kpis\tacc1_card8\t"$8}' | python _ce.py
