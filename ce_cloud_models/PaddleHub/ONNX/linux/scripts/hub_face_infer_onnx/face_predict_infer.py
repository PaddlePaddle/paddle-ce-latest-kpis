#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  face_predict_infer.py
  * @date  2021/5/24 11:05 AM
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

face_list0 = [
    'ultra_light_fast_generic_face_detector_1mb_640',
    'ultra_light_fast_generic_face_detector_1mb_320'
]
face_list1 = ['pyramidbox_lite_server_mask', 'pyramidbox_lite_mobile_mask']

for model in face_list0:
    res0 = paddle.load(
        os.path.join('infer_save', 'infer_face_model', model),
        model_filename='__model__',
        params_filename='__params__')
