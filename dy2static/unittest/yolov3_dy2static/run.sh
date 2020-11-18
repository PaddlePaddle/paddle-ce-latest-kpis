#!/bin/bash

model="yolov3"

log_path="./${model}.log"

pip3 install pycocotools

python3 download.py

python3 model.py --device=GPU --batch_size=4 --pass_num=2 > $log_path 2>&1
