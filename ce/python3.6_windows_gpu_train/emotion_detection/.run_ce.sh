#!/bin/bash
export FLAGS_enable_parallel_graph=1
export FLAGS_sync_nccl_allreduce=1
export FLAGS_fraction_of_gpu_memory_to_use=0.95
export CUDA_VISIBLE_DEVICES=0
python run_classifier.py --use_cuda true --do_train true --do_val true --epoch 2 --lr 0.002 --batch_size 64 --save_checkpoint_dir ./save_models/textcnn --save_steps 200 --validation_steps 200 --skip_steps 200 --enable_ce true | python _ce.py
			