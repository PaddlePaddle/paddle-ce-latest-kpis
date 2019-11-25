@echo off
set FLAGS_enable_parallel_graph=1
set FLAGS_sync_nccl_allreduce=1
set FLAGS_fraction_of_gpu_memory_to_use=0.95
set CUDA_VISIBLE_DEVICES=0
python run_classifier.py --task_name simnet --use_cuda true --do_train True --do_valid True --do_test True --do_infer False --batch_size 128 --train_data_dir ./data/train_pairwise_data --valid_data_dir ./data/test_pairwise_data --test_data_dir ./data/test_pairwise_data --infer_data_dir ./data/infer_data --output_dir ./model_files --config_path  ./config/bow_pairwise.json --vocab_path ./data/term2id.dict --epoch 1 --save_steps 10 --validation_steps 10 --compute_accuracy False --lamda 0.958 --task_mode pairwise --init_checkpoint ""  --enable_ce | python _ce.py 
				
