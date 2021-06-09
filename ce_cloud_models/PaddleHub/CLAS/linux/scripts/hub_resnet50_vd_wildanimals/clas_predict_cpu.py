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
classifier = hub.Module(name="resnet50_vd_wildanimals")
result = classifier.classification(
    images=[cv2.imread(img1), cv2.imread(img2)],
    batch_size=2,
    use_gpu=False,
    top_k=3)
expect = [{
    '其他': 0.14672206342220306,
    '老虎': 0.10811476409435272,
    '穿山甲': 0.09758924692869186
}, {
    '其他': 0.21444901823997498,
    '象牙制品': 0.08730071783065796,
    '穿山甲爪子': 0.08729685842990875
}]
width = classifier.get_expected_image_width()
height = classifier.get_expected_image_height()
mean = classifier.get_pretrained_images_mean()
std = classifier.get_pretrained_images_std()
for index in range(len(expect)):
    for key in result[index].keys():
        assert np.allclose(
            np.array(result[index][key]), np.array(expect[index][key]))
        # print(result[index][key])
        # print(np.array(expect[index][key]))
assert len(result) == 2
assert len(result[0]) == 3
assert len(result[1]) == 3
assert result[0].values() != []
assert width == 224
assert height == 224
assert np.allclose(np.array(mean), np.array([0.485, 0.456, 0.406]))
assert np.allclose(np.array(std), np.array([0.229, 0.224, 0.225]))
# print(result)
# print(mean)
# print(std)
# print(len(result))
# print(len(result[0]))
# print(result[0].values())
# print(sum(result[0].values()))
