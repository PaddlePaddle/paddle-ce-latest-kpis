#!/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf-8 vi:ts=4:sw=4:expandtab:ft=python
"""
/***************************************************************************
  *
  * Copyright (c) 2020 Baidu.com, Inc. All Rights Reserved
  * @file:  senta_predict_gpu.py
  * @date  2021/5/26 11:11 AM
  * @brief 
  *
  **************************************************************************/
"""
import paddlehub as hub

emot = hub.Module(name="emotion_detection_textcnn")
test_text = ["这家餐厅很好吃", "这部电影真的很差劲"]
expect0 = [{
    'text': ['这家餐厅很好吃', '这部电影真的很差劲'],
    'emotion_label': 1,
    'emotion_key': 'neutral',
    'positive_probs': 0.0087,
    'negative_probs': 0.0028,
    'neutral_probs': 0.9885
}, {
    'text': ['这家餐厅很好吃', '这部电影真的很差劲'],
    'emotion_label': 1,
    'emotion_key': 'neutral',
    'positive_probs': 0.0087,
    'negative_probs': 0.0028,
    'neutral_probs': 0.9885
}]
results0 = emot.emotion_classify(
    texts=[test_text, test_text], use_gpu=True, batch_size=2)
# print(results0)
assert expect0 == results0
expect1 = {'positive': 2, 'negative': 0, 'neutral': 1}
results1 = emot.get_labels()
# print(results1)
assert expect1 == results1
results2 = emot.get_vocab_path()
