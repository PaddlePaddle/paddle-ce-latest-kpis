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

face_landmark = hub.Module(name="openpose_body_estimation")

img2 = os.path.join(img_data, 'det_03.jpg')
result = face_landmark.predict(
    img=cv2.imread(img2), visualization=True, save_path=results)

assert len(result['candidate']) == 167
assert len(result['subset']) == 11
assert len(result['data']) == 342
