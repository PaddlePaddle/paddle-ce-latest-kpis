#!/bin/bash

model="seresnet"

log_path="./${model}.log"

python model.py --device=GPU --batch_size=8 --pass_num=5 --log_internal=10 > $log_path 2>&1
