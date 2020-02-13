#!/bin/bash

# This file is only used for continuous evaluation.

export ce_mode=1
python train.py --use_gpu=False --random_mirror=False --random_scaling=False 1> log
cat log | python _ce.py
