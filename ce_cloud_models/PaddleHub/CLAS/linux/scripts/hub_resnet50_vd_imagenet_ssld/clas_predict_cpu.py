#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  clas_predict_cpu.py
  * @date  2021/6/1 10:39 AM
  * @brief
  *
  **************************************************************************/
"""
import paddle
import paddlehub as hub
import cv2
import os
import shutil

paddle.set_device('cpu')
pwd = os.getcwd()
models_save = os.path.join(pwd, 'models_save')
pwd_last = os.path.abspath(os.path.join(os.getcwd(), ".."))
img_data = os.path.join(pwd_last, 'img_data')
results = os.path.join(pwd, 'results')
if os.path.exists(results):
    shutil.rmtree(results)

model = hub.Module(
    name='resnet50_vd_imagenet_ssld',
    label_list=["roses", "tulips", "daisy", "sunflowers", "dandelion"],
    load_checkpoint=os.path.join(models_save, 'best_model', 'model.pdparams'))
result = model.predict([os.path.join(img_data, 'image_flower.jpg')])
print(result)
assert result[0] != []
