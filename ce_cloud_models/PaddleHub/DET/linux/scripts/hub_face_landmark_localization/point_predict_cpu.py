#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  point_predict_cpu.py
  * @date  2021/5/25 10:56 AM
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

face_landmark = hub.Module(name="face_landmark_localization")
img1 = os.path.join(img_data, 'face_01.jpeg')
img2 = os.path.join(img_data, 'face_02.jpeg')
face_landmark.set_face_detector_module(
    hub.Module(name="ultra_light_fast_generic_face_detector_1mb_320"))
result = face_landmark.keypoint_detection(
    images=[cv2.imread(img1), cv2.imread(img2)],
    batch_size=2,
    use_gpu=False,
    visualization=True,
    output_dir=results)
result1 = face_landmark.keypoint_detection(
    paths=[img1, img2],
    batch_size=2,
    use_gpu=False,
    visualization=True,
    output_dir=results)
detector_module = face_landmark.get_face_detector_module()

assert len(result[0]['data'][0]) == 68
assert len(os.listdir(os.path.join(results))) == 4
for res, res1 in zip(result[0]['data'][0], result1[0]['data'][0]):
    assert np.allclose(np.array(res), np.array(res1))
