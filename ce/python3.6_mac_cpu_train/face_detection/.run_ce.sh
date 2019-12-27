#!/bin/bash

export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -u train.py --epoc_num=1 --use_multiprocess=False --batch_size=2 --pretrained_model=vgg_ilsvrc_16_fc_reduced --parallel=False --use_gpu False --enable_ce | python _ce.py

