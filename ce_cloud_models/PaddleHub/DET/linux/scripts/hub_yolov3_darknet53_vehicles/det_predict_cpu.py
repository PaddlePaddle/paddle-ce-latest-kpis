#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  det_predict_cpu.py.py
  * @date  2021/5/17 3:22 PM
  * @brief 
  *
  **************************************************************************/
"""
import paddlehub as hub
import cv2
import os
import shutil

pwd = os.getcwd()
models_save = os.path.join(pwd, 'models_save')
pwd_last = os.path.abspath(os.path.join(os.getcwd(), ".."))
img_data = os.path.join(pwd_last, 'img_data')
results = os.path.join(pwd, 'results')
if os.path.exists(results):
    shutil.rmtree(results)

image2 = os.path.join(img_data, 'det_02.jpeg')
image3 = os.path.join(img_data, 'det_03.jpg')
object_detector = hub.Module(name="yolov3_darknet53_vehicles")
result = object_detector.object_detection(
    images=[cv2.imread(image2), cv2.imread(image3)],
    output_dir=results,
    use_gpu=False,
    batch_size=2)
result_1 = object_detector.object_detection(
    paths=[image2, image3], output_dir=results, use_gpu=False, batch_size=2)

pic_list = [
    'det_02.jpeg', 'det_03.jpg', 'image_numpy_0.jpg', 'image_numpy_1.jpg'
]

for pic in pic_list:
    if not os.path.exists(os.path.join(results, pic)):
        raise Exception('result ' + pic + ' does not exist!!!')
