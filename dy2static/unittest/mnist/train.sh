#!/bin/bash

model="mnist"

log_path="./${model}.log"

python model.py --device=GPU --batch_size=128 --pass_num=5 > $log_path 2>&1 &