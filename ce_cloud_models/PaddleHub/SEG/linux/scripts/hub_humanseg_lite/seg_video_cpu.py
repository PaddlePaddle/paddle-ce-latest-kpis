#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  seg_video_cpu.py
  * @date  2021/5/24 2:30 PM
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

model = hub.Module(name='humanseg_lite')
vid = os.path.join(img_data, 'human_01.mp4')
model.video_segment(video_path=vid, use_gpu=False, save_dir=results)

assert len(os.listdir(os.path.join(results))) == 1
