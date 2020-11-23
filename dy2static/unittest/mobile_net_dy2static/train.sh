#!/bin/bash

model="mobile_net"
type="v1"

log_path="./${model}_${type}.log"

python model.py --model_type=$type --device=GPU --batch_size=32 --pass_num=20 --log_internal=10 > $log_path 2>&1
