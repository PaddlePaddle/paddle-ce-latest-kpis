#!/bin/bash

# This file is only used for continuous evaluation.
export FLAGS_cudnn_deterministic=True
export ce_mode=1
export CUDA_VISIBLE_DEVICES=0
python c_gan.py --batch_size=121 --epoch=1 --run_ce=True --use_gpu=True 1> log_cgan
cat log_cgan | python _ce.py
python dc_gan.py --batch_size=121 --epoch=1 --run_ce=True --use_gpu=True 1> log_dcgan
cat log_dcgan | python _ce.py


