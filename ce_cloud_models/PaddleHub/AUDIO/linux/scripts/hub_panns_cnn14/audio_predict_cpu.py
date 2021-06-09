#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  audio_predict_cpu.py
  * @date  2021/5/19 11:16 AM
  * @brief 
  *
  **************************************************************************/
"""
import argparse
import ast

import paddle

import paddlehub as hub
from paddlehub.datasets import ESC50
import numpy as np
from paddlehub.env import MODULE_HOME
import os

import librosa
import shutil

pwd = os.getcwd()
models_save = os.path.join(pwd, 'models_save')
pwd_last = os.path.abspath(os.path.join(os.getcwd(), ".."))
audio_data = os.path.join(pwd_last, 'audio_data')
results = os.path.join(pwd, 'results')
if os.path.exists(results):
    shutil.rmtree(results)

wav = os.path.join(audio_data, 'LJ001-0003.wav')  # 存储在本地的需要预测的wav文件
sr = 44100  # 音频文件的采样率
checkpoint = os.path.join(models_save, 'best_model',
                          'model.pdparams')  # 模型checkpoint
label_map = {idx: label for idx, label in enumerate(ESC50.label_list)}
model = hub.Module(
    name='panns_cnn14',
    task='sound-cls',
    num_class=ESC50.num_class,
    label_map=label_map,
    load_checkpoint=checkpoint)
data = [librosa.load(wav, sr=sr)[0]]
result = model.predict(
    data, sample_rate=sr, batch_size=1, feat_type='mel', use_gpu=False)
print(result[0])  # result[0]包含音频文件属于各类别的概率值
print('***' * 20)
topk = 10  # 展示音频得分前10的标签和分数
# 读取audioset数据集的label文件
label_file = os.path.join(MODULE_HOME, 'panns_cnn14', 'audioset_labels.txt')
label_map = {}
with open(label_file, 'r') as f:
    for i, l in enumerate(f.readlines()):
        label_map[i] = l.strip()
data = [librosa.load(wav, sr=sr)[0]]
result = model.predict(
    data, sample_rate=sr, batch_size=1, feat_type='mel', use_gpu=False)
msg = []
# 打印topk的类别和对应得分
for label, score in list(result[0].items())[:topk]:
    msg += f'{label}: {score}\n'
print(msg)
