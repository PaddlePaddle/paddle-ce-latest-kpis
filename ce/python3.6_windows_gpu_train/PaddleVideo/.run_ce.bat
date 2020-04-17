@echo off

set FLAGS_fast_eager_deletion_mode=1
set FLAGS_eager_delete_tensor_gb=0.0
set FLAGS_fraction_of_gpu_memory_to_use=0.98

set CUDA_VISIBLE_DEVICES=0
set sed="C:\Program Files\Git\usr\bin\sed.exe"

rem attention_cluster, out of GPU memory
%sed% -i s/"    num_gpus: 8"/"    num_gpus: 1"/g configs/attention_cluster.yaml
rem python train.py --model_name AttentionCluster --config=configs/attention_cluster.yaml --use_gpu=True  --fix_random_seed=True --epoch=1 >attention_cluster.log 
%sed% -i s/"    num_gpus: 8"/"    num_gpus: 1"/g configs/attention_lstm.yaml
rem python train.py --model_name AttentionLSTM --config=configs/attention_lstm.yaml --use_gpu=True  --fix_random_seed=True --epoch=1 >attention_lstm.log 

rem bmn
%sed% -i s/"    num_gpus: 4"/"    num_gpus: 1"/g configs/bmn.yaml
python train.py --model_name BMN --config=configs/bmn.yaml --use_gpu=True --batch_size=1 --fix_random_seed=True --epoch=1 >bmn.log 

rem bsn_tem
python train.py --model_name BSNTEM --config=configs/bsn_tem.yaml --use_gpu=True  --fix_random_seed=True --epoch=1 >bsn_tem.log
%sed% -i s/"    num_gpus: 8"/"    num_gpus: 1"/g configs/stnet.yaml 
python train.py --model_name STNET --config=configs/stnet.yaml --use_gpu=True  --fix_random_seed=True --epoch=1 > stnet.log 

rem tsm, out of GPU memory
%sed% -i s/"    num_gpus: 8"/"    num_gpus: 1"/g configs/tsm.yaml
rem python train.py --model_name=TSM --config=./configs/tsm.yaml --epoch=1 --fix_random_seed=True |python _ce.py
rem tsn, out of GPU memory 
%sed% -i s/"    num_gpus: 8"/"    num_gpus: 1"/g configs/tsn.yaml
rem python train.py --model_name=TSN --config=./configs/tsn.yaml --epoch=1 --fix_random_seed=True >tsn.log 






