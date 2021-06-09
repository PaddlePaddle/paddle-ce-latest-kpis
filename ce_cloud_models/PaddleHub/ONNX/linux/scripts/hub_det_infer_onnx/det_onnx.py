#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  det_onnx.py
  * @date  2021/5/11 5:24 PM
  * @brief 
  *
  **************************************************************************/
"""
import paddlehub as hub
import cv2
import shutil
import os

pwd = os.getcwd()
onnx_save = os.path.join(pwd, 'onnx_save')
if os.path.exists(onnx_save):
    shutil.rmtree(onnx_save)
os.mkdir(os.path.join(pwd, 'onnx_save'))

face_list = [
    'yolov3_resnet50_vd_coco2017', 'yolov3_resnet34_coco2017',
    'yolov3_mobilenet_v1_coco2017', 'yolov3_darknet53_vehicles',
    'yolov3_darknet53_pedestrian', 'yolov3_darknet53_coco2017',
    'ssd_vgg16_512_coco2017', 'ssd_vgg16_300_coco2017',
    'ssd_mobilenet_v1_pascal', 'faster_rcnn_resnet50_fpn_coco2017',
    'faster_rcnn_resnet50_coco2017'
]

not_exist_list = ['yolov3_darknet53_venus']

for model in face_list:
    face_detector = hub.Module(name=model)
    face_detector.export_onnx_model(
        dirname=os.path.join(onnx_save, 'onnx_det_model', face_detector.name),
        opset_version=12)
