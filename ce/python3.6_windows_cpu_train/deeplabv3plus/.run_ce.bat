@echo off

set MKL_NUM_THREADS=1
set OMP_NUM_THREADS=1
python ./train.py --batch_size=1 --train_crop_size=769 --total_step=50 --norm_type=gn --init_weights_path=./deeplabv3plus_gn_init --save_weights_path=model --dataset_path=./data/cityscape --use_multiprocessing false --use_gpu false --enable_ce | python _ce.py
				
