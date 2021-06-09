#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  seg_predict_infer.py
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

seg_list = ['humanseg_server', 'humanseg_mobile', 'humanseg_lite']

for model in seg_list:
    res = paddle.load(
        os.path.join('infer_save', 'infer_seg_model', model),
        model_filename='__model__',
        params_filename='__params__')
