#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  point_predict_gpu.py
  * @date  2021/5/25 11:53 AM
  * @brief 
  *
  **************************************************************************/
"""
import paddlehub as hub
import os
import cv2
import shutil
import numpy as np

pwd = os.getcwd()
models_save = os.path.join(pwd, 'models_save')
pwd_last = os.path.abspath(os.path.join(os.getcwd(), ".."))
img_data = os.path.join(pwd_last, 'img_data')
results = os.path.join(pwd, 'results')
if os.path.exists(results):
    shutil.rmtree(results)

face_landmark = hub.Module(name="human_pose_estimation_resnet50_mpii")
img1 = os.path.join(img_data, 'det_02.jpeg')
img2 = os.path.join(img_data, 'det_03.jpg')

result = face_landmark.keypoint_detection(
    images=[cv2.imread(img1), cv2.imread(img2)],
    batch_size=2,
    use_gpu=True,
    visualization=True,
    output_dir=results)
result1 = face_landmark.keypoint_detection(
    paths=[img1, img2],
    batch_size=2,
    use_gpu=True,
    visualization=True,
    output_dir=results)
