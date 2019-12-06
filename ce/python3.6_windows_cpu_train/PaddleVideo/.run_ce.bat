@echo off

set FLAGS_fast_eager_deletion_mode=1
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fraction_of_gpu_memory_to_use=0.98

set CUDA_VISIBLE_DEVICES=0
set sed="C:\Program Files\Git\usr\bin\sed.exe"
python train.py --model_name=TSM --config=./configs/tsm.yaml --epoch=1 --log_interval=10 --batch_size=1 --use_gpu False --fix_random_seed=True | python _ce.py




