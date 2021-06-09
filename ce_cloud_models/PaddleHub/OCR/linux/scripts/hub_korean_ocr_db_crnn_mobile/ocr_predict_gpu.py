#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  ocr_predict_cpu.py
  * @date  2021/5/10 4:15 PM
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
models_save = os.path.join(pwd, 'models_save')
pwd_last = os.path.abspath(os.path.join(os.getcwd(), ".."))
img_data = os.path.join(pwd_last, 'img_data')
results = os.path.join(pwd, 'results')
if os.path.exists(results):
    shutil.rmtree(results)

ocr = hub.Module(name="korean_ocr_db_crnn_mobile")
img0 = os.path.join(img_data, 'korean_word2.png')
result = ocr.recognize_text(
    images=[cv2.imread(img0)],
    use_gpu=True,
    visualization=True,
    output_dir=os.path.join(pwd, results))

for i in range(len(result[0]['data'])):
    print(result[0]['data'][i]['text'])
print(result)
assert len(result[0]['data']) == 29
assert len(os.listdir(os.path.join(results))) == 1
