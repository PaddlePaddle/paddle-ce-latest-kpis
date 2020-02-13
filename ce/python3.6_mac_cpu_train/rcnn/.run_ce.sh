#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1

python train.py --use_gpu False --model_save_dir=output/ --data_dir=dataexport/coco/ --max_iter=500 --enable_ce --pretrained_model=imagenet_resnet50_fusebn --learning_rate=0.00125 | python _ce.py


