#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  seg_predict_cpu.py
  * @date  2021/5/24 2:07 PM
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

pic_list = ['car.jpeg', 'det_03.jpg', 'small_bike.jpg', 'det_02.jpeg']

model = hub.Module(name='humanseg_server')
img0 = cv2.imread(os.path.join(img_data, pic_list[0]))
img1 = cv2.imread(os.path.join(img_data, pic_list[1]))
img2 = cv2.imread(os.path.join(img_data, pic_list[2]))
img3 = cv2.imread(os.path.join(img_data, pic_list[3]))
model.segment(
    images=[img0, img1, img2, img3],
    visualization=True,
    batch_size=4,
    use_gpu=False,
    output_dir=results)
model.segment(
    paths=[
        os.path.join(img_data, pic_list[0]),
        os.path.join(img_data, pic_list[1]),
        os.path.join(img_data, pic_list[2]),
        os.path.join(img_data, pic_list[3])
    ],
    visualization=True,
    batch_size=4,
    use_gpu=False,
    output_dir=results)

assert len(os.listdir(os.path.join(results))) == 2 * len(pic_list)
