@echo off

set CUDA_VISIBLE_DEVICES=0
set ce_mode=1
python ./train.py --batch_size=1 --train_crop_size=769 --total_step=50 --norm_type=gn --init_weights_path=./deeplabv3plus_gn_init --save_weights_path=model --dataset_path=./data/cityscape --use_multiprocessing false --use_gpu true --enable_ce | python _ce.py
				
