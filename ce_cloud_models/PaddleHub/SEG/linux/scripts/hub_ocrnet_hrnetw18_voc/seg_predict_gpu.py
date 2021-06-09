#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  seg_predict_gpu.py
  * @date  2021/5/8 4:23 PM
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

for pic in pic_list:
    model = hub.Module(
        name='ocrnet_hrnetw18_voc',
        pretrained=os.path.join(models_save, 'ocrnet_hrnetw18_voc', 'epoch_2',
                                'model.pdparams'))
    img = cv2.imread(os.path.join(img_data, pic))
    model.predict(images=[img], visualization=True, save_path=results)

assert len(os.listdir(os.path.join(results, 'image'))) == len(pic_list)
assert len(os.listdir(os.path.join(results, 'mask'))) == len(pic_list)
