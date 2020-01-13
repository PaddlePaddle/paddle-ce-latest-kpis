#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python tools/train.py -c configs/cascade_rcnn_r50_fpn_1x.yml --enable_ce=True -o max_iters=1000 FasterRCNNTrainFeed.shuffle=false CascadeBBoxAssigner.shuffle_before_sample=false FPNRPNHead.rpn_target_assign.use_random=false | python _ce.py
python tools/train.py -c configs/faster_rcnn_r50_fpn_1x.yml --enable_ce=True -o max_iters=1000 FasterRCNNTrainFeed.shuffle=false | python _ce.py
python tools/train.py -c configs/mask_rcnn_r50_fpn_1x.yml --enable_ce=True -o max_iters=1000 MaskRCNNTrainFeed.shuffle=false BBoxAssigner.shuffle_before_sample=false FPNRPNHead.rpn_target_assign.use_random=false | python _ce.py
python tools/train.py -c configs/yolov3_darknet.yml --enable_ce=True -o max_iters=200 YoloTrainFeed.shuffle=false YoloTrainFeed.use_process=false | python _ce.py



