#!/bin/bash
export FLAGS_enable_parallel_graph=1
export FLAGS_sync_nccl_allreduce=1
export FLAGS_fraction_of_gpu_memory_to_use=0.95

python run_classifier.py --task_name simnet --use_cuda false --do_train True --do_valid True --do_test False --do_infer False --batch_size 128 --train_data_dir ./data/zhidao --valid_data_dir ./data/zhidao --test_data_dir ./data/zhidao --infer_data_dir ./data/zhidao --output_dir ./model_files --config_path  ./config/bow_pairwise.json --vocab_path ./data/term2id.dict --epoch 6 --save_steps 1000 --validation_steps 1 --compute_accuracy True --lamda 0.958 --task_mode pairwise --init_checkpoint ""  --enable_ce | python _ce.py				
				
