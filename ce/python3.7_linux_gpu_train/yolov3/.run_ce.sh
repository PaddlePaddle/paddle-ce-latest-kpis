#!/bin/bash
#This file is only used for continuous evaluation.
export CUDA_VISIBLE_DEVICES=0
python train.py --enable_ce True --use_multiprocess False --snapshot_iter 100 --max_iter 200 1> log_1card
cat log_1card | python _ce.py
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python train.py --enable_ce True --use_multiprocess False --snapshot_iter 100 --max_iter 200 1> log_8cards
cat log_8cards | python _ce.py

#eval
export CUDA_VISIBLE_DEVICES=0
python eval.py --dataset=coco2017 --weights=weights/yolov3 1>yolov3_eval
if [ $? -ne 0 ];then
    echo -e "yolov3,eval,FAIL"
else
    echo -e "yolov3,eval,SUCCESS"
fi
#infer
python infer.py --dataset=coco2017 --weights=weights/yolov3 --image_path=dataset/coco/val2017 --image_name=000000000139.jpg --draw_thresh=0.5 >yolov3_infer
if [ $? -ne 0 ];then
    echo -e "yolov3,infer,FAIL"
else
    echo -e "yolov3,infer,SUCCESS"
fi

