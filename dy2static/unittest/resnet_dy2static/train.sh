#!/bin/bash

model="resnet"

log_path="./${model}.log"

python model.py --device=GPU --batch_size=8 --pass_num=20 --log_internal=10 > $log_path 2>&1
