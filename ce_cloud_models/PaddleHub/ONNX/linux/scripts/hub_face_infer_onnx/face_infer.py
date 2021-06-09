#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  face_infer.py
  * @date  2021/5/24 11:04 AM
  * @brief 
  *
  **************************************************************************/
"""
import paddlehub as hub
import cv2
import shutil
import os

pwd = os.getcwd()
infer_save = os.path.join(pwd, 'infer_save')
if os.path.exists(infer_save):
    shutil.rmtree(infer_save)
os.mkdir(os.path.join(pwd, 'infer_save'))

face_list = [
    'ultra_light_fast_generic_face_detector_1mb_640',
    'ultra_light_fast_generic_face_detector_1mb_320',
    'pyramidbox_lite_server_mask', 'pyramidbox_lite_mobile_mask'
]

for model in face_list:
    face_detector = hub.Module(name=model)
    face_detector.save_inference_model(dirname=os.path.join(
        infer_save, 'infer_face_model', face_detector.name))
