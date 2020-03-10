#!/bin/bash
export FLAGS_enable_parallel_graph=1
export FLAGS_sync_nccl_allreduce=1
export FLAGS_fraction_of_gpu_memory_to_use=0.95
export ce_mode=1
python run_classifier.py --use_cuda false --do_train true --do_val true --epoch 10 --lr 0.002 --batch_size 64 --save_checkpoint_dir ./save_models/textcnn --save_steps 200 --validation_steps 200 --skip_steps 200 --random_seed 90 --enable_ce true | python _ce.py
			