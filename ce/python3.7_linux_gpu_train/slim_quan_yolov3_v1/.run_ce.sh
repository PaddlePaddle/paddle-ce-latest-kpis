#!/bin/bash

# This file is only used for continuous evaluation.

export ce_mode=1
rm -rf *_factor.txt

export FLAGS_eager_delete_tensor_gb=0.0
export CUDA_VISIBLE_DEVICES=4,5,6,7
export current_dir=$PWD
cd ${current_dir}/slim/quantization
if [ -d 'checkpoints' ]; then
    rm -rf checkpoints
fi
sed -i "s/end_epoch: 4/end_epoch: 1/g" yolov3_mobilenet_v1_slim.yaml
sed -i "s/epoch: 5/epoch: 1/g" yolov3_mobilenet_v1_slim.yaml
python compress.py \
    -s yolov3_mobilenet_v1_slim.yaml  \
    -c ../../configs/yolov3_mobilenet_v1_voc.yml \
    -d "../../dataset/voc" \
    -o max_iters=258 \
    LearningRate.base_lr=0.0001 \
    LearningRate.schedulers='[!PiecewiseDecay {gamma: 0.1, milestones: [258, 516]}]' \
    pretrain_weights=https://paddlemodels.bj.bcebos.com/object_detection/yolov3_mobilenet_v1_voc.tar \
    YoloTrainFeed.batch_size=64 >${current_dir}/yolov3_v1_quan_run.log 2>&1

cd ${current_dir}
cat yolov3_v1_quan_run.log|grep mAP | awk -F " " 'END{print "kpis\ttest_mAP\t"$6}' | python _ce.py
