#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  porn_predict_cpu.py
  * @date  2021/5/26 11:11 AM
  * @brief 
  *
  **************************************************************************/
"""
import paddlehub as hub

porn = hub.Module(name="porn_detection_lstm")
test_text = ["黄片下载", "打击黄牛党"]
expect0 = [{
    'text': '黄片下载',
    'porn_detection_label': 1,
    'porn_detection_key': 'porn',
    'porn_probs': 0.9879,
    'not_porn_probs': 0.0121
}, {
    'text': '打击黄牛党',
    'porn_detection_label': 0,
    'porn_detection_key': 'not_porn',
    'porn_probs': 0.0004,
    'not_porn_probs': 0.9996
}]
results0 = porn.detection(texts=test_text, use_gpu=False, batch_size=2)
# print(results0)
assert expect0 == results0
expect1 = {'porn': 1, 'not_porn': 0}
results1 = porn.get_labels()
# print(results1)
assert expect1 == results1
results2 = porn.get_vocab_path()
