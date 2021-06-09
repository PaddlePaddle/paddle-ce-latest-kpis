#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  seg_infer.py
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

seg_list = ['humanseg_server', 'humanseg_mobile', 'humanseg_lite']

for model in seg_list:
    seg_detector = hub.Module(name=model)
    seg_detector.save_inference_model(dirname=os.path.join(
        infer_save, 'infer_seg_model', seg_detector.name))
