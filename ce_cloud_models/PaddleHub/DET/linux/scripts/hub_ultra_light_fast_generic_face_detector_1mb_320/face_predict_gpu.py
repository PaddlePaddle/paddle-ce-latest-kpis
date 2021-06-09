#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  face_predict_cpu.py
  * @date  2021/5/24 10:33 AM
  * @brief 
  *
  **************************************************************************/
"""
import paddlehub as hub
import os
import cv2
import shutil

pwd = os.getcwd()
models_save = os.path.join(pwd, 'models_save')
pwd_last = os.path.abspath(os.path.join(os.getcwd(), ".."))
img_data = os.path.join(pwd_last, 'img_data')
results = os.path.join(pwd, 'results')
if os.path.exists(results):
    shutil.rmtree(results)

face_detector = hub.Module(
    name="ultra_light_fast_generic_face_detector_1mb_320")
result = face_detector.face_detection(
    images=[
        cv2.imread(os.path.join(img_data, 'face_01.jpeg')),
        cv2.imread(os.path.join(img_data, 'face_02.jpeg'))
    ],
    output_dir=results,
    batch_size=2,
    use_gpu=True,
    visualization=True)

result_1 = face_detector.face_detection(
    paths=[
        os.path.join(img_data, 'face_01.jpeg'),
        os.path.join(img_data, 'face_02.jpeg')
    ],
    output_dir=results,
    batch_size=2,
    use_gpu=True,
    visualization=True)

assert len(os.listdir(os.path.join(results))) == 4
