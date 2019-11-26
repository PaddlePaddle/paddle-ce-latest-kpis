@echo off
rem This file is only used for continuous evaluation.

set MKL_NUM_THREADS=1
set OMP_NUM_THREADS=1
python -u train.py --use_multiprocess=False --batch_size=64 --dataset=pascalvoc --pretrained_model=pretrained/ssd_mobilenet_v1_coco/ --epoc_num=1 --model_save_dir=output_pascalvoc --use_gpu False --enable_ce=True  | python _ce.py
