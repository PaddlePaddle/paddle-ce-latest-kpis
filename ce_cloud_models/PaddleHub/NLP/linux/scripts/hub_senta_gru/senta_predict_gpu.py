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

senta = hub.Module(name="senta_gru")
test_text = ["这家餐厅很好吃", "这部电影真的很差劲"]
expect0 = [{
    'text': ['这家餐厅很好吃', '这部电影真的很差劲'],
    'sentiment_label': 1,
    'sentiment_key': 'positive',
    'positive_probs': 0.754,
    'negative_probs': 0.246
}, {
    'text': ['这家餐厅很好吃', '这部电影真的很差劲'],
    'sentiment_label': 1,
    'sentiment_key': 'positive',
    'positive_probs': 0.754,
    'negative_probs': 0.246
}]
results0 = senta.sentiment_classify(
    texts=[test_text, test_text], use_gpu=True, batch_size=2)
# print(results0)
assert expect0 == results0
expect1 = {'positive': 1, 'negative': 0}
results1 = senta.get_labels()
assert expect1 == results1
results2 = senta.get_vocab_path()
