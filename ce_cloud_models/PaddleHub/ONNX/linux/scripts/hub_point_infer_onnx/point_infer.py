#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  point_infer.py
  * @date  2021/5/24 11:48 AM
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

point_list1 = [
    'face_landmark_localization', 'human_pose_estimation_resnet50_mpii'
]
point_list2 = ['openpose_body_estimation']

for model in point_list1:
    point_detector = hub.Module(name=model)
    point_detector.save_inference_model(dirname=os.path.join(
        infer_save, 'infer_point_model', point_detector.name))

for model in point_list2:
    point_detector = hub.Module(name=model)
    point_detector.save_inference_model(save_dir=os.path.join(
        infer_save, 'infer_point_model', point_detector.name))
