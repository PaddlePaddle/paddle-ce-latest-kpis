#!/bin/bash
# cascade_rcnn
python tools/train.py -c configs/cascade_rcnn_r50_fpn_1x.yml -o max_iters=100 use_gpu=false FasterRCNNTrainFeed.shuffle=false CascadeBBoxAssigner.shuffle_before_sample=false FPNRPNHead.rpn_target_assign.use_random=false
# faster_rcnn
sed -i s/"base_lr: 0.02"/"base_lr: 0.0025"/g configs/faster_rcnn_r50_fpn_1x.yml
python tools/train.py -c configs/faster_rcnn_r50_fpn_1x.yml -o max_iters=100 use_gpu=false FasterRCNNTrainFeed.shuffle=false
# mask_rcnn
sed -i s/"base_lr: 0.01"/"base_lr: 0.00125"/g configs/mask_rcnn_r50_fpn_1x.yml
python tools/train.py -c configs/mask_rcnn_r50_fpn_1x.yml -o max_iters=100 use_gpu=false MaskRCNNTrainFeed.shuffle=false BBoxAssigner.shuffle_before_sample=false FPNRPNHead.rpn_target_assign.use_random=false
# yolov3
python tools/train.py -c configs/yolov3_darknet.yml -o max_iters=100 use_gpu=false YoloTrainFeed.shuffle=false YoloTrainFeed.use_process=false



