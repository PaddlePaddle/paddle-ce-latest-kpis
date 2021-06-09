#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  det_predict_onnx_cpu.py
  * @date  2021/5/12 3:27 PM
  * @brief 
  *
  **************************************************************************/
"""
import paddlehub as hub
import paddle
import numpy as np
import cv2
import os
import onnxruntime as rt
import shutil

pwd = os.getcwd()
onnx_save = os.path.join(pwd, 'onnx_save')
pwd_last = os.path.abspath(os.path.join(os.getcwd(), ".."))
img_data = os.path.join(pwd_last, 'img_data')


def randtool(dtype, low, high, shape):
    """
    np random tools
    """
    if dtype == "int":
        return np.random.randint(low, high, shape)

    elif dtype == "float":
        return low + (high - low) * np.random.random(shape)


face_list = [
    'yolov3_resnet50_vd_coco2017', 'yolov3_resnet34_coco2017',
    'yolov3_mobilenet_v1_coco2017', 'yolov3_darknet53_vehicles',
    'yolov3_darknet53_pedestrian', 'yolov3_darknet53_coco2017',
    'ssd_vgg16_512_coco2017', 'ssd_vgg16_300_coco2017',
    'ssd_mobilenet_v1_pascal'
]

inputnum_list = [2, 2, 2, 2, 2, 2, 1, 1, 1]
inputshape_list = [608, 608, 608, 608, 608, 608, 512, 300, 300]

for model, input_num, input_shape in zip(face_list, inputnum_list,
                                         inputshape_list):
    inputs_dict = {}
    sess = rt.InferenceSession(
        os.path.join(onnx_save, 'onnx_det_model', model, model + '.onnx'))
    input_data1 = randtool("float", -1.0, 1.0,
                           (1, 3, input_shape, input_shape)).astype(np.float32)
    inputs_dict[sess.get_inputs()[0].name] = input_data1
    if input_num == 2:
        input_data2 = np.array([[input_shape, input_shape]]).astype(np.int32)
        inputs_dict[sess.get_inputs()[1].name] = input_data2
    onnx_result = sess.run(None, input_feed=inputs_dict)
    print(onnx_result[0])
