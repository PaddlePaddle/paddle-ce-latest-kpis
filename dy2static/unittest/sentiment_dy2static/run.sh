#!/bin/bash

model="sentiment"

log_path="./${model}.log"

python model.py --device=GPU --batch_size=4 --pass_num=5 > $log_path 2>&1
