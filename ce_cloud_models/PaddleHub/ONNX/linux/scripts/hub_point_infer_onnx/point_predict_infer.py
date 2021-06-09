#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  point_predict_infer.py
  * @date  2021/5/24 4:28 PM
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

point_list1 = ['face_landmark_localization']
point_list2 = ['human_pose_estimation_resnet50_mpii']
point_list3 = ['openpose_body_estimation']

for model in point_list2:
    res0 = paddle.load(
        os.path.join('infer_save', 'infer_point_model', model),
        model_filename='__model__',
        params_filename='__params__')
