#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  det_predict_infer_cpu.py
  * @date  2021/5/12 3:27 PM
  * @brief 
  *
  **************************************************************************/
"""
import paddlehub as hub
import paddle
import cv2
import os
import shutil

pwd = os.getcwd()
infer_save = os.path.join(pwd, 'infer_save')
pwd_last = os.path.abspath(os.path.join(os.getcwd(), ".."))
img_data = os.path.join(pwd_last, 'img_data')

face_list = [
    'yolov3_resnet50_vd_coco2017', 'yolov3_resnet34_coco2017',
    'yolov3_mobilenet_v1_coco2017', 'yolov3_darknet53_vehicles',
    'yolov3_darknet53_pedestrian', 'yolov3_darknet53_coco2017',
    'ssd_vgg16_512_coco2017', 'ssd_vgg16_300_coco2017',
    'ssd_mobilenet_v1_pascal', 'faster_rcnn_resnet50_fpn_coco2017',
    'faster_rcnn_resnet50_coco2017'
]
len_list = [387, 291, 241, 366, 366, 366, 79, 71, 199, 189, 169]

for model, res_len in zip(face_list, len_list):
    res = paddle.load(
        os.path.join('infer_save', 'infer_det_model', model),
        model_filename='__model__',
        params_filename='__params__')
    assert len(res.keys()) == res_len
