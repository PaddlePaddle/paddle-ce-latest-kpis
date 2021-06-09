#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  clas_predict_cpu.py
  * @date  2021/5/27 2:23 PM
  * @brief 
  *
  **************************************************************************/
"""
import paddle
import paddlehub as hub
import cv2
import os
import shutil
import numpy as np

pwd = os.getcwd()
models_save = os.path.join(pwd, 'models_save')
pwd_last = os.path.abspath(os.path.join(os.getcwd(), ".."))
img_data = os.path.join(pwd_last, 'img_data')
results = os.path.join(pwd, 'results')
if os.path.exists(results):
    shutil.rmtree(results)

img1 = os.path.join(img_data, 'cls_01.jpg')
img2 = os.path.join(img_data, 'seg_01.jpeg')
classifier = hub.Module(name="se_resnext101_32x4d_imagenet")
input_dict = {"image": [img1, img2]}
result = classifier.classification(data=input_dict)
expect = [[{
    'lion': 0.9973996877670288
}], [{
    'sulphur-crested cockatoo': 0.31934070587158203
}]]

assert np.allclose(
    np.array(result[0][0]['lion']), np.array(expect[0][0]['lion']))
assert np.allclose(
    np.array(result[1][0]['sulphur-crested cockatoo']),
    np.array(expect[1][0]['sulphur-crested cockatoo']))
# print(result)
