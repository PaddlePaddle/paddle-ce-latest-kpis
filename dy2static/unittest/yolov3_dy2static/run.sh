#!/bin/bash

model="yolov3"

log_path="./${model}.log"

pip install pycocotools

python download.py

python model.py --device=GPU --batch_size=4 --pass_num=2 > $log_path 2>&1
