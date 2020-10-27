#!/bin/bash

model="reinforcement_learning"

log_path="./${model}.log"

python model.py --device=GPU --pass_num=360 --gamma=0.99 --log_interval=10 > $log_path 2>&1
