@echo off
rem This file is only used for continuous evaluation.

set CUDA_VISIBLE_DEVICES=0
python -u train.py --use_multiprocess=False --batch_size=64 --dataset=pascalvoc --pretrained_model=pretrained/ssd_mobilenet_v1_coco/ --epoc_num=1 --model_save_dir=output_pascalvoc --enable_ce=True --use_gpu True | python _ce.py
