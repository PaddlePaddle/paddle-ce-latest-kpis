#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  seg_video_stream_cpu.py
  * @date  2021/5/24 2:30 PM
  * @brief 
  *
  **************************************************************************/
"""
import cv2
import numpy as np
import paddlehub as hub
import os
import shutil

pwd = os.getcwd()
models_save = os.path.join(pwd, 'models_save')
pwd_last = os.path.abspath(os.path.join(os.getcwd(), ".."))
img_data = os.path.join(pwd_last, 'img_data')
results = os.path.join(pwd, 'results')
if os.path.exists(results):
    shutil.rmtree(results)

human_seg = hub.Module(name='humanseg_mobile')
vid = os.path.join(img_data, 'human_01.mp4')
cap_video = cv2.VideoCapture(vid)
fps = cap_video.get(cv2.CAP_PROP_FPS)
save_path = os.path.join(results, 'humanseg_mobile_video.avi')
width = int(cap_video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap_out = cv2.VideoWriter(save_path,
                          cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps,
                          (width, height))
prev_gray = None
prev_cfd = None
while cap_video.isOpened():
    ret, frame_org = cap_video.read()
    if ret:
        [img_matting, prev_gray, prev_cfd] = human_seg.video_stream_segment(
            frame_org=frame_org,
            frame_id=cap_video.get(1),
            prev_gray=prev_gray,
            prev_cfd=prev_cfd)
        img_matting = np.repeat(img_matting[:, :, np.newaxis], 3, axis=2)
        bg_im = np.ones_like(img_matting) * 255
        comb = (img_matting * frame_org +
                (1 - img_matting) * bg_im).astype(np.uint8)
        cap_out.write(comb)
    else:
        break

cap_video.release()
cap_out.release()
