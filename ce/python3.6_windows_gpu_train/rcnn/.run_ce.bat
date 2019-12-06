@echo off

set MKL_NUM_THREADS=1
set OMP_NUM_THREADS=1

set CUDA_VISIBLE_DEVICES=0
python train.py --model_save_dir=output/ --data_dir=dataset/coco/ --max_iter=500 --enable_ce --pretrained_model=imagenet_resnet50_fusebn --learning_rate=0.00125 | python _ce.py


