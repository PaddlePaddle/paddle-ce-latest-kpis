@echo off
rem train
set OPENBLAS_NUM_THREADS=1 
set CPU_NUM=1
python train.py --train_data_dir data/convert_text8 --dict_path data/test_build_dict --num_passes 1 --batch_size 100 --model_output_dir v1_cpu5_b100_lr1dir --base_lr 1.0 --print_batch 1000 --with_speed --is_sparse 


