#!/bin/bash

export FLAGS_fast_eager_deletion_mode=1
export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98

export CUDA_VISIBLE_DEVICES=0
python train.py --use_multiprocess=False --snapshot_iter 100 --max_iter 200 >log_1card 2>&1
cat log_1card | grep "Iter" | tail "-1" | tr ',' ' ' | awk '{print "kpis\tloss_card1\t"$4}' | python _ce.py

cat log_1card | grep "Iter" | tail "-1" | tr ',' ' '| awk '{print "kpis\ttime_card1\t"$6}' | python _ce.py

export CUDA_VISIBLE_DEVICES=0,1,2,3
python -m paddle.distributed.launch --gpus=0,1,2,3 train.py --use_multiprocess=False --use_data_parallel=1 --batch_size=8 --snapshot_iter 100 --max_iter 200 >log_4card 2>&1
cat log_4card | grep "Iter" | tail "-1" | tr ',' ' ' | awk '{print "kpis\tloss_card4\t"$4}' | python _ce.py

cat log_4card | grep "Iter" | tail "-1" | tr ',' ' '| awk '{print "kpis\ttime_card4\t"$6}' | python _ce.py

